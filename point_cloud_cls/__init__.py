import logging.config

from .log_config import LOGGER_CONFIG
from .transform import TrainTransform, BaseTransform
from .ldgcnn import SimpleClsLDGCN

logging.config.dictConfig(LOGGER_CONFIG)
