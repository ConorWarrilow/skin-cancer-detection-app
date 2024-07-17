import os
import sys
import logging
from datetime import datetime



LOG_DIR = "logs"
LOG_FILENAME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, F"{LOG_FILENAME}.log")


logging.basicConfig(
    level= logging.INFO,
    format= "[%(asctime)s] %(lineno)d || %(levelname)s || %(module)s || %(message)s",

    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("cancer-logger")