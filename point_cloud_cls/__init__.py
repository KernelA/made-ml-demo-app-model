import logging.config

from .log_config import LOGGER_CONFIG
from .lighting_model import ClassificationModelTrainer

logging.config.dictConfig(LOGGER_CONFIG)
