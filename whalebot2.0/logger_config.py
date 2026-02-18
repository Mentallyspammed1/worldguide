import logging
import os
from logging.handlers import RotatingFileHandler

def setup_custom_logger(name, log_directory="bot_logs", level=logging.INFO):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = RotatingFileHandler(
            os.path.join(log_directory, f"{name}.log"),
            maxBytes=10*1024*1024,
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
