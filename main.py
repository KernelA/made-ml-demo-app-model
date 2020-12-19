import argparse
import configargparse
import pathlib
import logging
import json
from typing import List, Optional
import random

import torch
from torch.nn import parameter
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


def train_one_epoch(device, data_loader: gdata.DataLoader, model: SimpleClsLDGCN, optimizer) -> List[float]:
    model.train()
    losses = []

    true = []
    pred = []
    i = 0
    for point_cloud_batch in tqdm.tqdm(data_loader):
        optimizer.zero_grad()
        gpu_point_cloud = point_cloud_batch.to(device)
        prediction = model(gpu_point_cloud)
        loss = loss_function(prediction, gpu_point_cloud.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        true.extend(point_cloud_batch.y.tolist())
        pred.extend(prediction.argmax(dim=1).cpu().tolist())
        i += 1
        if i > 5:
            break

    optimizer.zero_grad()

    return losses, accuracy_score(true, pred)


def save_training_state(epoch, model, optimizer, checkpoint_dir: pathlib.Path):
    LOGGER.info("Save state %s to %s", epoch, checkpoint_dir)
    latest_dir = checkpoint_dir / LATEST_DIR_NAME
    latest_dir.mkdir(exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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

    i = 0
    with torch.no_grad():
        for test_point_cloud in test_loader:
            output = model(test_point_cloud.to(device)).argmax(dim=1)
            true.extend(test_point_cloud.y.tolist())
            pred.extend(output.tolist())
            i += 1
            if i > 5:
                pass

    return classification_report(true, pred, target_names=test_loader.dataset.label_encoder.classes_,
                                 output_dict=True, zero_division=0), confusion_matrix(true, pred)


def save_report(epoch: int, report_dir: pathlib.Path, report: dict):
    LOGGER.info("Save report of %s to %s", epoch, report_dir)
    with open(report_dir / f"test_report_{epoch}.json", "w", encoding="utf-8") as report_file:
        json.dump(report, report_file)


def save_conf_matrix(epoch: int, report_dir: pathlib.Path, conf_matrix: np.ndarray):
    LOGGER.info("Save confusion matrix of %s to %s", epoch, report_dir)
    np.savetxt(str(report_dir / f"conf_matrix_{epoch}.csv"), conf_matrix, fmt="%d", delimiter=",", encoding="utf-8")


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
        LOGGER.warning("CUDA is not available.")

    device = torch.device(device_name)
    LOGGER.info("Select device: %s", device)

    data_params = train_param.DataParams()
    model_params = train_param.ModelParams()
    optim_params = train_param.OptimizerParams()

    num_points = data_params.num_points
    num_features = model_params.in_chan

    base_transform = BaseTransform(num_points)
    train_transform = TrainTransform(num_points, angle_degree=15, axis=2, rnd_shift=0.02)

    train_dataset = dataset.SubsampleModelNet40(args.data_root_dir, transform=train_transform, train=True)

    LOGGER.info("Train model on %s features to classify %s classes", num_features, train_dataset.num_classes)

    model_params_dict = {"out_chan":  train_dataset.num_classes, **model_params.__dict__}
    del model_params
    model = SimpleClsLDGCN(**model_params_dict).to(device)
    del model_params_dict

    LOGGER.info("Transfer model to %s", device)

    test_dataset = dataset.SubsampleModelNet40(args.data_root_dir, transform=base_transform, train=False)

    train_loader = gdata.DataLoader(train_dataset, batch_size=data_params.batch_size,
                                    pin_memory=True, num_workers=args.num_workers, shuffle=True)
    test_loader = gdata.DataLoader(test_dataset, batch_size=data_params.batch_size,
                                   pin_memory=True, num_workers=args.num_workers)

    adamw = optim.AdamW(model.parameters(), **optim_params.__dict__)

    LOGGER.info("Try to load state from %s", check_dir)
    state = load_state(check_dir)

    if state is None:
        LOGGER.warning("Cannot load latest state for resume training. Begin from zero")
    else:
        LOGGER.info("Found previous state. Try load and resume training")
        model.load_state_dict(state["model_state_dict"])
        adamw.load_state_dict(state["optimizer_state_dict"])
        epoch = state["epoch"]

    with SummaryWriter(str(log_dir), comment="Train classification model", flush_secs=60) as log_writer:
        LOGGER.info("Begin training")

        for epoch in tqdm.trange(data_params.epochs):
            losses, accuracy = train_one_epoch(device, train_loader, model, adamw)
            total_loss = sum(losses) / len(losses)
            LOGGER.info("Epoch: %d Loss: %f.4  Train accuracy: %f.2", epoch, total_loss, accuracy)
            log_writer.add_scalar("Train/loss cross entropy", total_loss, global_step=epoch)
            log_writer.add_scalar("Train/overall accuracy", accuracy, global_step=epoch)

            if epoch % args.save_every == 0:
                save_training_state(epoch, model, adamw, check_dir)
                save_model_state(epoch, model, check_dir)

            if epoch % args.test_every == 0:
                cls_report, conf_matrix = test_model(device, model, test_loader)
                total = conf_matrix.sum()
                accuracy = np.trace(conf_matrix) / total
                log_writer.add_scalar("Test/overall accuracy", accuracy, global_step=epoch)
                acc_per_class = {label: conf_matrix[index][index] / total
                                 for index, label in enumerate(test_loader.dataset.label_encoder.classes_)}
                log_writer.add_scalars("Test/accuracy per class", acc_per_class, global_step=epoch)
                save_report(epoch, rep_dir, cls_report)
                save_conf_matrix(epoch, rep_dir, conf_matrix)
                LOGGER.info("Epoch: %d Test accuracy: %f.2", epoch, total_loss, accuracy)

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

    args = parser.parse_args()

    main(args)
