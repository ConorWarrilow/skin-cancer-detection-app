import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from CancerClassifier import logger
#from CancerClassifier.config import 
from CancerClassifier.components.datapreparation import DataPreparation
from CancerClassifier.pipeline.datapreparationpipe import DataPreparationPipeline
from CancerClassifier.config import ConfigurationManager

STAGE_NAME = "Data Preparation Stage"


try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = DataPreparationPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e