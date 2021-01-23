"""Python logging config

"""

_BASE_FORMAT = '%(asctime)-s %(levelname)-s %(message)s'


LOGGER_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'simple_console_format': {
            'format': _BASE_FORMAT
        }
    },
    'handlers': {
        'default_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple_console_format',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {
        "lightning": {
            'level': 'INFO',
            'handlers': ['default_console']
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['default_console']
    }
}
