import os
import shutil
import logging
from typing import List

import torch
import gdown

from torch_geometric import data, io
from sklearn.preprocessing import LabelEncoder


class SubsampleModelNet40(data.InMemoryDataset):
    URL = "https://drive.google.com/uc?id=1fRcBvYY5oFYSd71-O9bUhvc8FSO74aJc"

    CLASSES = frozenset((
        "airplane",
        "bathtub",
        "bed",
        "bench",
        "bookshelf",
        "bottle",
        "bowl",
        "car",
        "chair",
        "cone",
        "cup",
        "curtain",
        "desk",
        "door",
        "dresser",
        "flower_pot",
        "glass_box",
        "guitar",
        "keyboard",
        "lamp",
        "laptop",
        "mantel",
        "monitor",
        "night_stand",
        "person",
        "piano",
        "plant",
        "radio",
        "range_hood",
        "sink",
        "sofa",
        "stairs",
        "stool",
        "table",
        "tent",
        "toilet",
        "tv_stand",
        "vase",
        "wardrobe",
        "xbox")
    )

    def __init__(self, root, classes: List[str], train: bool = True, transform=None, pre_transform=None):
        for class_name in classes:
            if class_name not in self.CLASSES:
                raise ValueError(f"Class '{class_name} not in ModelNet40")

        self.logger = logging.getLogger()
        self.classes = classes
        self.train = train
        self.data_index = int(train)
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(self.raw_file_names)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[self.data_index])

    @ property
    def raw_file_names(self):
        return self.classes

    @ property
    def label_encoder(self):
        return self._label_encoder

    @ property
    def processed_file_names(self):
        return ["test.pth", "train.pth"]

    def download(self):
        self.logger.debug("Download from %s to %s", self.URL, self.raw_dir)
        path_to_archive = os.path.join(self.raw_dir, "ModelNet40.zip")

        if not os.path.isfile(path_to_archive):
            gdown.download(self.URL, path_to_archive)
        else:
            self.logger.info("ZIp already exist. Skip downloading")

        data_dir = os.path.join(self.raw_dir, "ModelNet40")
        if not os.path.isdir(data_dir):
            self.logger.debug("Extract zip %s to %s", path_to_archive, self.raw_dir)
            data.extract_zip(path_to_archive, self.raw_dir)

        with os.scandir(data_dir) as entry_it:
            for entry in entry_it:
                if entry.is_dir() and entry.name in self.raw_file_names:
                    os.rename(entry.path, os.path.join(self.raw_dir, entry.name))

        os.remove(path_to_archive)
        self.logger.debug("Remove %s", data_dir)
        shutil.rmtree(data_dir)

    def process(self):
        for prefix, processed_path in zip(("test", "train"), self.processed_file_names):
            data_list = []
            with os.scandir(self.raw_dir) as entry_it:
                for entry in entry_it:
                    if entry.is_dir() and entry.name in self.raw_file_names:
                        off_dir = os.path.join(entry.path, prefix)
                        for off_file in os.listdir(off_dir):
                            if os.path.splitext(off_file)[1] == ".off":
                                mesh = io.read_off(os.path.join(off_dir, off_file))
                                mesh.y = torch.from_numpy(self._label_encoder.transform([entry.name]))
                                data_list.append(mesh)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            self.logger.info("Save to %s", processed_path)
            torch.save((data, slices), os.path.join(self.processed_dir, processed_path))


class SimpleShapes(data.InMemoryDataset):
    CLASSES = frozenset((
        "cone",
        "cube",
        "cylinder",
        "plane",
        "torus",
        "uv_sphere"
    )
    )
    URL = "https://drive.google.com/uc?id=1AZAwX7FtYZPWN3uIeiWmIdT9sRFpw67A"

    def __init__(self, root: str, classes: List[str], train: bool = True, transform=None, pre_transform=None, pre_filter=None):
        for class_name in classes:
            if class_name not in self.CLASSES:
                raise ValueError(f"Class '{class_name} not in datset")

        self.logger = logging.getLogger()
        self.classes = classes
        self.train = train
        self.data_index = int(train)
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(self.raw_file_names)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.data_index])

    @ property
    def raw_file_names(self):
        return self.classes

    @ property
    def label_encoder(self):
        return self._label_encoder

    @ property
    def processed_file_names(self):
        return ["test.pth", "train.pth"]

    def download(self):
        self.logger.debug("Download from %s to %s", self.URL, self.raw_dir)
        path_to_archive = os.path.join(self.raw_dir, "shapes.zip")

        if not os.path.isfile(path_to_archive):
            gdown.download(self.URL, path_to_archive)
        else:
            self.logger.info("Zip already exist. Skip downloading")

        self.logger.debug("Extract zip %s to %s", path_to_archive, self.raw_dir)
        data.extract_zip(path_to_archive, self.raw_dir)
        os.remove(path_to_archive)

    def process(self):
        for prefix, processed_path in zip(("test", "train"), self.processed_file_names):
            data_list = []
            with os.scandir(self.raw_dir) as entry_it:
                for entry in entry_it:
                    if entry.is_dir() and entry.name in self.raw_file_names:
                        off_dir = os.path.join(entry.path, prefix)
                        for off_file in os.listdir(off_dir):
                            if os.path.splitext(off_file)[1] == ".obj":
                                mesh = io.read_obj(os.path.join(off_dir, off_file))
                                mesh.y = torch.from_numpy(self._label_encoder.transform([entry.name]))
                                data_list.append(mesh)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            self.logger.info("Save to %s", processed_path)
            torch.save((data, slices), os.path.join(self.processed_dir, processed_path))
