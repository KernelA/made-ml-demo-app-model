import logging.config

from .log_config import LOGGER_CONFIG
from .transform import TrainTransform, BaseTransform
from .lighting_model import ClassificationModelTrainer

logging.config.dictConfig(LOGGER_CONFIG)
