import configargparse
import pathlib
import logging
import json
from typing import List, Optional

import torch
import tqdm
from torch import optim, nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric import data as gdata
from torch_geometric import datasets as gdatasets
from sklearn.metrics import classification_report

from point_cloud_cls import SimpleClsLDGCN, BaseTransform, TrainTransform


LOGGER = logging.getLogger()

LATEST_DIR_NAME = "latest"


def init():
    torch.backends.cudnn.benchmark = True


def loss_function(predicted: torch.Tensor, true: torch.Tensor):
    return nn.functional.nll_loss(predicted, true, reduction="sum")


def train_one_epoch(device, data_loader: gdata.DataLoader, model: SimpleClsLDGCN, optimizer) -> List[float]:
    model.train()
    losses = []
    i = 0
    for point_cloud_batch in tqdm.tqdm(data_loader):
        optimizer.zero_grad()
        gpu_point_cloud = point_cloud_batch.to(device)
        prediction = model(gpu_point_cloud)
        loss = loss_function(prediction, gpu_point_cloud.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        i += 1
        if i > 10:
            break

    optimizer.zero_grad()

    return losses


def save_state(epoch, model, optimizer, checkpoint_dir: pathlib.Path):
    LOGGER.info("Save state %s to %s", epoch, checkpoint_dir)
    latest_dir = checkpoint_dir / LATEST_DIR_NAME
    latest_dir.mkdir(exist_ok=True)

    for file in latest_dir.iterdir():
        if file.is_file() and file.name.endswith(".pth"):
            file.rename(latest_dir.parents[1] / file.name)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, str(latest_dir / f"training_state_{epoch}.pth"))


def load_state(checkpoint_dir: pathlib.Path) -> Optional[dict]:
    latest_checkpoint = checkpoint_dir / LATEST_DIR_NAME
    if latest_checkpoint.exists():
        for file in (checkpoint_dir / LATEST_DIR_NAME).iterdir():
            if file.is_file() and file.name.endswith(".pth"):
                state = torch.load(str(file))
                return state
    return None


@ torch.no_grad()
def test_model(device, model, test_loader):
    LOGGER.info("Test model")
    model.eval()

    true = []
    pred = []

    i = 0
    for test_point_cloud in test_loader:
        output = model(test_point_cloud.to(device)).argmax(dim=1)
        true.extend(test_point_cloud.y.tolist())
        pred.extend(output.tolist())
        i += 1
        if i > 10:
            break

    return classification_report(true, pred, target_names=test_loader.dataset.raw_file_names,
                                 output_dict=True, labels=tuple(range(len(test_loader.dataset.raw_file_names))))


def save_report(epoch: int, report_dir: pathlib.Path, report: dict):
    LOGGER.info("Save report of %s to %s", epoch, report_dir)
    with open(report_dir / f"test_report_{epoch}.json", "w", encoding="utf-8") as report_file:
        json.dump(report, report_file)


def train(args):
    exp_dir = pathlib.Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    check_dir = exp_dir / "checkpoints"
    check_dir.mkdir(exist_ok=True)

    rep_dir = exp_dir / "test_reports"
    rep_dir.mkdir(exist_ok=True)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    LOGGER.info("Select device: %s", device)

    base_transform = BaseTransform(args.num_points)
    train_transform = TrainTransform(args.num_points, 15, 2, 0.02)

    train_dataset = gdatasets.ModelNet(args.data_root_dir, transform=train_transform, train=True)

    LOGGER.info("Train model on %s features to classify %s classes", args.num_features, train_dataset.num_classes)

    model = SimpleClsLDGCN(args.num_features, train_dataset.num_classes).to(device)
    LOGGER.info("Transfer model to %s", device)

    test_dataset = gdatasets.ModelNet(args.data_root_dir, transform=base_transform, train=False)

    train_loader = gdata.DataLoader(train_dataset, batch_size=args.batch_size,
                                    pin_memory=True, num_workers=args.num_workers, shuffle=True)
    test_loader = gdata.DataLoader(test_dataset, batch_size=args.batch_size,
                                   pin_memory=True, num_workers=args.num_workers)

    adamw = optim.AdamW(model.parameters(), lr=1e-2)

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

        for epoch in tqdm.trange(args.epochs):
            losses = train_one_epoch(device, train_loader, model, adamw)
            total_loss = sum(losses) / len(losses)
            log_writer.add_scalar("Cross entropy", total_loss, global_step=epoch)

            if epoch % args.save_every == 0:
                save_state(epoch, model, adamw, check_dir)

            if epoch % args.test_every == 0:
                cls_report = test_model(device, model, test_loader)
                save_report(epoch, rep_dir, cls_report)

        LOGGER.info("End training")


def main(args):
    train(args)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, required=True, help="Train config")
    parser.add_argument("--batch_size", default=64, type=int, help="A batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="A number of epochs")
    parser.add_argument("--data_root_dir", type=str, required=True, help="A path to data dir")
    parser.add_argument("--exp_dir", required=True, type=str, help="A path to directory with checkpoints and logs")
    parser.add_argument("--num_workers", default=2, type=int, help="A number of workers to load data")
    parser.add_argument("--num_points", default=2048, type=int, help="A number of points to sample data")
    parser.add_argument("--num_features", type=int, default=3, help="A number of features")
    parser.add_argument("--checkpoint_every_epoch", dest="save_every", type=int,
                        default=1, help="Save checkpoint after each epoch")
    parser.add_argument("--test_every", type=int,
                        default=1, help="Save checkpoint after each epoch")

    args = parser.parse_args()

    main(args)
