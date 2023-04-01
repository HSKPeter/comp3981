# import logging
# from datetime import datetime
# import os
# import uuid
#
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
#
# now = datetime.now()
# formatted_date_time = now.strftime('%Y-%m-%d_%H%M')
#
# package_directory = os.path.dirname(os.path.abspath(__file__))
# log_file_path = os.path.join(package_directory, "log", f"{formatted_date_time}_exploration_{uuid.uuid4().hex}.log")
#
# file_handler = logging.FileHandler(log_file_path, mode='a')
# file_handler.setLevel(logging.INFO)
# file_handler.flush = file_handler.stream.flush
# formatter = logging.Formatter('%(asctime)s - %(levelname)s\n %(message)s\n')
# file_handler.setFormatter(formatter)
#
# logger.addHandler(file_handler)
from loguru import logger
import sys

logger.remove()  # Remove all configured default handlers

fmt="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n<level>{message}</level>\n\n"

logger.add(sys.stderr, format=fmt)
