from loguru import logger
import sys

logger.remove()  # Remove all configured default handlers

log_format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n<level>{message}</level>\n\n"

logger.add(sys.stderr, format=log_format)
