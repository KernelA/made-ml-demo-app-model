import argparse
import configargparse
import pathlib
import logging
import json
from typing import List, Optional
import random

import torch
import tqdm
import yaml
import numpy as np
from torch import optim, nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric import data as gdata
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from point_cloud_cls import SimpleClsLDGCN, BaseTransform, TrainTransform, dataset
import train_param


LOGGER = logging.getLogger()

LATEST_DIR_NAME = "latest"
CUDA_DEVICE = "cuda"
CPU_DEVICE = "cpu"


def init(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.seed()
    np.random.seed(seed)
    random.seed(seed)


def loss_function(predicted: torch.Tensor, true: torch.Tensor):
    return nn.functional.nll_loss(predicted, true, reduction="sum")


def train_one_epoch(epoch: int, device, data_loader: gdata.DataLoader,
                    model: SimpleClsLDGCN, optimizer, writer) -> List[float]:
    model.train()
    losses = []

    true = []
    pred = []

    colors = None

    for point_cloud_batch in tqdm.tqdm(data_loader):
        optimizer.zero_grad()
        gpu_point_cloud = point_cloud_batch.to(device)
        prediction = model(gpu_point_cloud)
        loss = loss_function(prediction, gpu_point_cloud.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        true_in_batch = gpu_point_cloud.y
        pred_in_batch = prediction.argmax(dim=1)
        incorrect = (true_in_batch != pred_in_batch).cpu()

        if incorrect.any():
            incorrect_index = random.choice(torch.nonzero(incorrect).view(-1))
            vertices = point_cloud_batch.pos[point_cloud_batch.batch == incorrect_index, :].unsqueeze(0)

            if colors is None:
                colors = torch.zeros_like(vertices, dtype=torch.uint8)
                colors[:, :, 0] = 255

            true_label = data_loader.dataset.label_encoder.inverse_transform(true_in_batch[None, incorrect_index].cpu())
            pred_label = data_loader.dataset.label_encoder.inverse_transform(pred_in_batch[None, incorrect_index].cpu())
            writer.add_mesh(f"Train/incorrect_classify_expected_{true_label}_predicted_{pred_label}",
                            vertices=vertices, colors=colors,
                            global_step=epoch)

        true.extend(true_in_batch.tolist())
        pred.extend(pred_in_batch.tolist())

    optimizer.zero_grad()

    return losses, accuracy_score(true, pred)


def save_training_state(epoch: int, model, optimizer, scheduler, checkpoint_dir: pathlib.Path):
    LOGGER.info("Save state %s to %s", epoch, checkpoint_dir)
    latest_dir = checkpoint_dir / LATEST_DIR_NAME
    latest_dir.mkdir(exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, str(latest_dir / f"training_state_latest.pth"))


def save_model_state(epoch, model, checkpoint_dir: pathlib.Path):
    LOGGER.info("Save state %s to %s", epoch, checkpoint_dir)
    torch.save(model.state_dict(),  str(checkpoint_dir / f"model_state_{epoch}.pth"))


def load_state(checkpoint_dir: pathlib.Path) -> Optional[dict]:
    latest_checkpoint = checkpoint_dir / LATEST_DIR_NAME
    if latest_checkpoint.exists():
        for file in (checkpoint_dir / LATEST_DIR_NAME).iterdir():
            if file.is_file() and file.name.endswith(".pth"):
                state = torch.load(str(file))
                return state
    return None


def test_model(device, model, test_loader):
    LOGGER.info("Test model")
    model.eval()

    true = []
    pred = []

    with torch.no_grad():
        for test_point_cloud in test_loader:
            output = model(test_point_cloud.to(device)).argmax(dim=1)
            true.extend(test_point_cloud.y.tolist())
            pred.extend(output.tolist())

    return classification_report(true, pred, target_names=test_loader.dataset.label_encoder.classes_,
                                 output_dict=True, zero_division=0), confusion_matrix(true, pred)


def save_report(epoch: int, report_dir: pathlib.Path, report: dict):
    LOGGER.info("Save report of %s to %s", epoch, report_dir)
    with open(report_dir / f"test_report_{epoch}.json", "w", encoding="utf-8") as report_file:
        json.dump(report, report_file)


def save_conf_matrix(epoch: int, report_dir: pathlib.Path, conf_matrix: np.ndarray):
    LOGGER.info("Save confusion matrix of %s to %s", epoch, report_dir)
    np.savetxt(str(report_dir / f"conf_matrix_{epoch}.csv"), conf_matrix,
               fmt="%d", delimiter=",", encoding="utf-8")


def load_yaml(path_to_file):
    with open(path_to_file, "r", encoding="utf-8") as yml_config:
        return yaml.safe_load(yml_config)


def train(args):
    exp_dir = pathlib.Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    check_dir = exp_dir / "checkpoints"
    check_dir.mkdir(exist_ok=True)

    rep_dir = exp_dir / "test_reports"
    rep_dir.mkdir(exist_ok=True)

    device_name = CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE

    if device_name != CUDA_DEVICE:
        LOGGER.warning("CUDA is not available")

    device = torch.device(device_name)
    LOGGER.info("Select device: %s", device)

    data_params = train_param.DataParams()
    model_params = train_param.ModelParams()
    optim_params = train_param.OptimizerParams()

    num_points = data_params.num_points
    num_features = model_params.in_chan

    base_transform = BaseTransform(num_points)
    train_transform = TrainTransform(num_points, angle_degree=15, axis=2, rnd_shift=0.02)

    classes = load_yaml(args.data_config)["classes"]

    train_dataset = dataset.SubsampleModelNet40(args.data_root_dir, classes, transform=train_transform, train=True)

    LOGGER.info("Train model on %s features to classify %s classes", num_features, train_dataset.num_classes)

    model_params_dict = {"out_chan":  train_dataset.num_classes, **model_params.__dict__}
    del model_params
    model = SimpleClsLDGCN(**model_params_dict).to(device)
    del model_params_dict

    LOGGER.info("Transfer model to %s", device)

    test_dataset = dataset.SubsampleModelNet40(args.data_root_dir, classes, transform=base_transform, train=False)

    train_loader = gdata.DataLoader(train_dataset, batch_size=data_params.batch_size,
                                    pin_memory=True, num_workers=args.num_workers, shuffle=True)
    test_loader = gdata.DataLoader(test_dataset, batch_size=data_params.batch_size,
                                   pin_memory=True, num_workers=args.num_workers)

    scheduler_params = train_param.SchedulerParams()

    optimizer = optim.AdamW(model.parameters(), **optim_params.__dict__)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, data_params.epochs, **scheduler_params.__dict__)

    LOGGER.info("Try to load state from %s", check_dir)
    state = load_state(check_dir)

    if state is None:
        LOGGER.warning("Cannot load latest state for resume training. Begin from zero")
        epoch = 0
    else:
        LOGGER.info("Found previous state. Try load and resume training")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler"])
        epoch = state["epoch"]

    with SummaryWriter(str(log_dir), comment="Train classification model", flush_secs=60) as log_writer:
        LOGGER.info("Begin training")

        for epoch in tqdm.tqdm(range(epoch, data_params.epochs)):
            losses, accuracy = train_one_epoch(epoch, device, train_loader, model, optimizer, log_writer)
            total_loss = sum(losses) / len(losses)
            LOGGER.info("Epoch: %d Loss: %.4f  Train accuracy: %.2f", epoch, total_loss, accuracy)
            log_writer.add_scalar("Train/loss cross entropy", total_loss, global_step=epoch)
            log_writer.add_scalar("Train/overall accuracy", accuracy, global_step=epoch)

            if epoch % args.save_every == 0:
                save_training_state(epoch, model, optimizer, scheduler, check_dir)
                save_model_state(epoch, model, check_dir)

            if epoch % args.test_every == 0:
                cls_report, conf_matrix = test_model(device, model, test_loader)
                total = conf_matrix.sum()
                accuracy = np.trace(conf_matrix) / total
                log_writer.add_scalar("Test/overall accuracy", accuracy, global_step=epoch)
                total_samples_per_class = conf_matrix.sum(axis=1)
                acc_per_class = {label: conf_matrix[index][index] / total_samples_per_class[index]
                                 for index, label in enumerate(test_loader.dataset.label_encoder.classes_)}
                log_writer.add_scalars("Test/accuracy per class", acc_per_class, global_step=epoch)
                save_report(epoch, rep_dir, cls_report)
                save_conf_matrix(epoch, rep_dir, conf_matrix)
                LOGGER.info("Epoch: %d Test accuracy: %.2f", epoch, accuracy)

            log_writer.add_scalar("Train/learning rate", np.array(scheduler.get_last_lr()), global_step=epoch)
            scheduler.step()

        LOGGER.info("End training")


def main(args):
    init(train_param.SEED)
    train(args)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, required=True, help="Train config")
    parser.add_argument("--data_root_dir", type=str, required=True, help="A path to data dir")
    parser.add_argument("--exp_dir", required=True, type=str, help="A path to directory with checkpoints and logs")
    parser.add_argument("--num_workers", default=2, type=int, help="A number of workers to load data")
    parser.add_argument("--checkpoint_every_epoch", dest="save_every", type=int,
                        default=1, help="Save checkpoint after each epoch")
    parser.add_argument("--test_every", type=int,
                        default=1, help="Save checkpoint after each epoch")
    parser.add_argument("--data_config", required=True, type=str, help="A path to data config")

    args = parser.parse_args()

    main(args)
