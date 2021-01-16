import os
import shutil
import logging
from typing import List
import glob

import py7zr
import torch
from torch_geometric import data, io
from sklearn.preprocessing import LabelEncoder


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

    def __init__(self, path_to_data: str, root: str,
                 classes: List[str], is_train: bool = True, transform=None, pre_transform=None, pre_filter=None):
        for class_name in classes:
            if class_name not in self.CLASSES:
                raise ValueError(f"Class '{class_name} not in datset")

        self.logger = logging.getLogger()
        self._path_to_data = path_to_data
        self.classes = set(classes)
        self.is_train = is_train
        self.data_index = int(is_train)
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(tuple(self.classes))
        super().__init__(root, transform, pre_transform, pre_filter)
        self.logger.info("Load data from %s", self.processed_paths[self.data_index])
        self.data, self.slices = torch.load(self.processed_paths[self.data_index])

    @ property
    def raw_file_names(self):
        return self._find_archive()

    @ property
    def label_encoder(self):
        return self._label_encoder

    @ property
    def processed_file_names(self):
        return ["test.pth", "train.pth"]

    def download(self):
        if not os.path.isfile(self._path_to_data):
            raise ValueError(f"Cannot find '{self._path_to_data}")

        self.logger.debug("Copy from %s to %s", self._path_to_data, self.raw_dir)
        shutil.copy2(self._path_to_data, self.raw_dir)

    def _find_archive(self) -> List[str]:
        return glob.glob(os.path.join(self.raw_dir, "*.7z"))

    def process(self):
        path_to_archive = self._find_archive()[0]

        with py7zr.SevenZipFile(path_to_archive, "r") as archive:
            self.logger.debug("Extract all data to %s", self.raw_dir)
            archive.extractall(self.raw_dir)

        for prefix, processed_path in zip(("test", "train"), self.processed_file_names):
            data_list = []
            with os.scandir(self.raw_dir) as entry_it:
                for entry in entry_it:
                    if entry.is_dir() and entry.name in self.classes:
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

        with os.scandir(self.raw_dir) as entry_it:
            for entry in entry_it:
                if entry.name != os.path.basename(path_to_archive):
                    if entry.is_dir():
                        self.logger.debug("Remove dir '%s'", entry.path)
                        shutil.rmtree(entry.path)
                    elif entry.is_file():
                        self.logger.debug("Remove file '%s'", entry.path)
                        os.remove(entry.path)
