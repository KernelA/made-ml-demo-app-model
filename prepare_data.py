
import configargparse

from point_cloud_cls import dataset


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, required=True, help="Train config")
    parser.add_argument("--data_root_dir", type=str, required=True, help="A path to data dir")
    parser.add_argument("--classes", action="append", required=True, help="A path to data dir")
    args = parser.parse_args()

    # Download only and save to torch format
    _ = dataset.SubsampleModelNet40(args.data_root_dir, args.classes)