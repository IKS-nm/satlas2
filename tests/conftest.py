import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
