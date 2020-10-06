#-*- encoding=utf8 -*-
from __future__ import absolute_import
import logging

#global logger
logger = None #logging.getLogger()

def init_logger(log_file = None , log_file_level = logging.NOTSET):
    log_format = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != "":
        file_hands = logging.FileHandler(log_file,mode="w+")
        file_hands.setLevel(log_file_level)
        file_hands.setFormatter(log_format)
        logger.addFilter(file_hands)

    return logger