import configargparse
import pathlib
import logging
from typing import List

import yaml
import torch
from torch_geometric import data
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from nn_model import SimpleClsLDGCN, BaseTransform, TrainTransform
from point_cloud_cls import ClassificationModelTrainer, dataset
import train_param


LOGGER = logging.getLogger()


def init(seed: int):
    seed_everything(seed)


def load_yaml(path_to_file):
    with open(path_to_file, "r", encoding="utf-8") as yml_config:
        return yaml.safe_load(yml_config)


def get_model(class_labels: List[str]):
    model_params = train_param.ModelParams()
    model_params_dict = {"out_chan":  len(class_labels), **model_params.__dict__}
    model = SimpleClsLDGCN(**model_params_dict)
    optim_params = train_param.OptimizerParams()
    scheduler_params = train_param.SchedulerParams()

    LOGGER.info("Train model on %s features to classify %s classes", model_params.in_chan, len(class_labels))

    lighting_model = ClassificationModelTrainer(
        model, optim_params, scheduler_params, class_labels)

    return lighting_model


def get_train_loader(dataset, batch_size: int, num_workers: int):
    return data.DataLoader(dataset, batch_size=batch_size,
                           pin_memory=True, num_workers=num_workers, shuffle=True)


def get_test_loader(dataset, batch_size: int, num_workers: int):
    return data.DataLoader(dataset, batch_size=batch_size,
                           pin_memory=True, num_workers=num_workers)


def get_train_test_dataset(num_points: int, data_config):
    base_transform = BaseTransform(num_points)
    train_transform = TrainTransform(num_points, angle_degree=15, axis=2, rnd_shift=0.02)

    data_root = data_config["data_root_dir"]
    classes = data_config["classes"]
    input_archive = data_config["input"]

    train_dataset = dataset.SimpleShapes(input_archive, data_root, classes, transform=train_transform, is_train=True)
    test_dataset = dataset.SimpleShapes(input_archive, data_root, classes, transform=base_transform, is_train=False)

    return train_dataset, test_dataset


def train(args):
    exp_dir = pathlib.Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    check_dir = exp_dir / "checkpoints"
    check_dir.mkdir(exist_ok=True)

    data_config = load_yaml(args.data_config)

    data_params = train_param.DataParams()

    train_dataset, test_dataset = get_train_test_dataset(data_params.num_points, data_config)

    train_params = train_param.TrainParams()

    train_loader = get_train_loader(train_dataset, train_params.batch_size, args.num_workers)
    test_loader = get_test_loader(test_dataset, train_params.batch_size, args.num_workers)
    model = get_model(train_dataset.label_encoder.classes_)

    checkpoint_callback = ModelCheckpoint(monitor="Train/overall_accuracy",
                                          save_top_k=3,
                                          period=5,
                                          mode="max",
                                          filename="{epoch}-{Train/overall_accuracy:.2f}")

    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")

    early_stopping = EarlyStopping(monitor="Test/overall_accuracy", min_delta=1e-2,
                                   mode="max", verbose=True, strict=False)

    gpus = 1

    if not torch.cuda.is_available():
        LOGGER.warning("CUDA is not available. Fallback to CPU training. It may be very slow")
        gpus = None

    trainer = Trainer(gpus=gpus,
                      deterministic=train_params.deterministic,
                      benchmark=train_params.benchmark,
                      check_val_every_n_epoch=train_params.valid_every,
                      default_root_dir=str(exp_dir),
                      fast_dev_run=args.fast_dev_run,
                      max_epochs=train_params.epochs,
                      callbacks=[learning_rate_monitor, checkpoint_callback, early_stopping])

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=test_loader)


def main(args):
    init(train_param.SEED)
    train(args)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, required=True, help="Train config")
    parser.add_argument("--exp_dir", required=True, type=str, help="A path to directory with checkpoints and logs")
    parser.add_argument("--num_workers", default=2, type=int, help="A number of workers to load data")
    parser.add_argument("--data_config", required=True, type=str, help="A path to data config")
    parser.add_argument("--fast_dev_run", action="store_true", help="Enable fast dev run for testing purpose")

    args = parser.parse_args()

    main(args)
