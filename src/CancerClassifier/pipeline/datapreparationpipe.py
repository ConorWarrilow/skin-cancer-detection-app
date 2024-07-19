import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from CancerClassifier import logger
from CancerClassifier.config import ConfigurationManager
from CancerClassifier.components.datapreparation import DataPreparation


STAGE_NAME = "Data Preparation Stage"

class DataPreparationPipeline():
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            data_config = config_manager.initiate_data_configuration()
            dataprep = DataPreparation(data_config)
            dataprep.get_transformed_data()
        except Exception as e:
            raise e
        

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = DataPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e