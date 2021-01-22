import pathlib
import logging
import json
import pickle

import configargparse
import torch

from point_cloud_cls import dataset, BaseTransform
from train_param import ModelParams, DataParams
from train import load_yaml

PICKLE_PROTOCOL = 4

LOGGER = logging.getLogger()


def export_model(model_dir, path_to_checkpoint, num_classes: int):
    model_params = ModelParams()
    model_params_dict = {"out_chan":  num_classes, **model_params.__dict__}

    path_to_dump = model_dir / "model_params.json"

    with open(model_dir / "model_params.json", "w", encoding="utf-8") as config_file:
        json.dump(model_params_dict, config_file)

    LOGGER.info("Save model config to %s", path_to_dump)

    checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
    model_state = checkpoint["state_dict"]

    replace_prefix = "cls_model."
    new_state = dict()

    for key in model_state:
        new_key = key
        if key.startswith(replace_prefix):
            new_key = key[len(replace_prefix):]
        new_state[new_key] = model_state[key]

    path_to_copy_checkpoint = model_dir / "model_state.pth"

    torch.save(new_state, path_to_copy_checkpoint)
    LOGGER.info("Save model state to %s", path_to_checkpoint)


def export_label_encoder(label_encoder_dir, label_encoder):
    path_to_dump = label_encoder_dir / "label_encoder.pickle"
    with open(label_encoder_dir / "label_encoder.pickle", "wb") as label_encoder_file:
        pickle.dump(label_encoder, label_encoder_file, protocol=PICKLE_PROTOCOL)

    LOGGER.info("Save label encoder to %s", path_to_dump)


def main(args):
    export_dir = pathlib.Path(args.export_dir)
    export_dir.mkdir(exist_ok=True, parents=True)
    LOGGER.info("Save all files to %s", export_dir)

    data_config = args.data_config

    num_points = DataParams().num_points

    data_config = load_yaml(data_config)
    data_root = data_config["data_root_dir"]
    classes = data_config["classes"]

    base_transform = BaseTransform(num_points)
    test_dataset = dataset.SimpleShapes(data_config["input"], data_root,
                                        classes, transform=base_transform, is_train=False)

    model_data_dir = export_dir / "model"
    model_data_dir.mkdir(exist_ok=True)
    export_model(model_data_dir, args.checkpoint, test_dataset.num_classes)
    label_encoder_dir = export_dir / "label_encoder"
    label_encoder_dir.mkdir(exist_ok=True)
    export_label_encoder(label_encoder_dir, test_dataset.label_encoder)


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add_argument("--config", is_config_file=True, required=True, help="A path to config")
    parser.add_argument("--data_config", dest="data_config", required=True,
                        type=str, help="A path to prepare_data config")
    parser.add_argument("--checkpoint", required=True, type=str, help="A path to model checkpoint")
    parser.add_argument("--export_dir", required=True, type=str, help="A path to directory with all files")
    parser.add_argument("--device", required=True, choices=["cpu", "cuda"], help="A device type to run model")

    args = parser.parse_args()
    main(args)
