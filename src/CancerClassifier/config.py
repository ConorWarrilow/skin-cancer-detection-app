import os
from dataclasses import dataclass
from pathlib import Path
from CancerClassifier.constants import *
from CancerClassifier.utils import read_yaml, create_directories
#from CancerClassifier.utils import ISICDatasetTrain, ISICDatasetValid
from params import AUGMENTATIONS



# Frozen means we can't add any more functionality to the dataclass after it's created
#@dataclass(frozen = True)
#class DataIngestionConfig:
#    root_dir: Path
#    source_URL: str
#    local_data_file: Path
#    unzip_dir: Path
#
#
#@dataclass(frozen=True)
#class PrepareBaseModelConfig:
#    root_dir: Path
#    base_model_path: Path
#    updated_base_model_path: Path
#    params_image_size: list
#    params_learning_rate: float
#    params_include_top: bool
#    params_weights: str
#    params_classes: int
#
#
#@dataclass(frozen=True)
#class TrainingConfig:
#    root_dir: Path
#    trained_model_path: Path
#    updated_base_model_path: Path
#    training_data: Path
#    params_epochs: int
#    params_batch_size: int
#    params_is_augmentation: bool
#    params_image_size: list
#
#
#@dataclass(frozen=True)
#class EvaluationConfig:
#    path_of_model: Path
#    training_data: Path
#    all_params: dict
#    mlflow_uri: str
#    params_image_size: list
#    params_batch_size: int
#
#
#
#class ConfigurationManager:
#    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
#
#        # Setting the main 
#        self.config = read_yaml(config_filepath)
#        self.params = read_yaml(params_filepath)
#        create_directories([self.config.artifacts_root]) # creating artifacts directory
#
#
#
#    def get_data_ingestion_config(self) -> DataIngestionConfig:
#        config = self.config.data_ingestion
#        create_directories([config.root_dir]) # Creating root directory which is in the data ingestion key
#
#        data_ingestion_config = DataIngestionConfig(root_dir = config.root_dir,
#                                                    source_URL = config.source_URL,
#                                                    local_data_file = config.local_data_file,
#                                                    unzip_dir = config.unzip_dir)
#        
#        return data_ingestion_config
#    
#
#
#    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
#        config = self.config.prepare_base_model
#        
#        create_directories([config.root_dir])
#
#        prepare_base_model_config = PrepareBaseModelConfig(
#            root_dir=Path(config.root_dir), # maybe we should add Path() to the other config we did
#            base_model_path=Path(config.base_model_path),
#            updated_base_model_path=Path(config.updated_base_model_path),
#            params_image_size=self.params.IMAGE_SIZE,
#            params_learning_rate=self.params.LEARNING_RATE,
#            params_include_top=self.params.INCLUDE_TOP,
#            params_weights=self.params.WEIGHTS,
#            params_classes=self.params.CLASSES
#        )
#
#        return prepare_base_model_config
#    
#
#
#    def get_training_config(self) -> TrainingConfig:
#        training = self.config.training
#        prepare_base_model = self.config.prepare_base_model
#        params = self.params
#        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest-CT-Scan-data")
#        create_directories([
#            Path(training.root_dir)
#        ])
#
#        training_config = TrainingConfig(
#            root_dir=Path(training.root_dir),
#            trained_model_path=Path(training.trained_model_path),
#            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
#            training_data=Path(training_data),
#            params_epochs=params.EPOCHS,
#            params_batch_size=params.BATCH_SIZE,
#            params_is_augmentation=params.AUGMENTATION,
#            params_image_size=params.IMAGE_SIZE
#        )
#
#        return training_config
#    
#
#    def get_evaluation_config(self) -> EvaluationConfig:
#        eval_config = EvaluationConfig(
#            path_of_model = "artifacts/training/model.h5",
#            training_data = "artifacts/data_ingestion/Chest-CT-Scan-data",
#            mlflow_uri = "https://dagshub.com/ConorWarrilow/cancer-classifier-app.mlflow",
#            all_params = self.params,
#            params_image_size = self.params.IMAGE_SIZE,
#            params_batch_size = self.params.BATCH_SIZE
#        )
#        return eval_config
    








@dataclass(frozen=True)
class DataConfiguration:
    root_dir: Path
    train_images_path: Path
    train_csv_path: Path
    test_images_path: Path
    test_csv_path: Path
    base_fold: int
    n_folds: int


@dataclass(frozen=True)
class DataLoaderConfiguration:
    train_batch_size: int
    valid_batch_size: int
    img_size: int
    augmentations: dict


@dataclass(frozen=True)
class BaseModelConfiguration:
    root_dir: Path
    base_models_path: Path
    model_name: str
    checkpoint_path: Path


@dataclass(frozen=True)
class ModelTrainingConfiguration:
    root_dir: Path
    trained_model_path: Path
    scheduler: str
    epochs: int
    learning_rate: float
    min_lr: float
    weight_decay: float
    t_max: int
    n_accumulate: int



class ConfigurationManager():
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH, augmentations_dict = AUGMENTATIONS):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.augmentations = augmentations_dict
        
    def initiate_data_configuration(self) -> DataConfiguration:
        config = self.config.data_config
        params = self.params

        data_configuration = DataConfiguration(root_dir = config.root_dir,
                                               train_images_path = config.train_images_path,
                                               train_csv_path = config.train_csv_path,
                                               test_images_path = config.test_images_path,
                                               test_csv_path = config.test_csv_path,
                                               base_fold = self.params.BASE_FOLD,
                                               n_folds = self.params.N_FOLDS,
                                               )
        return data_configuration
    

    def initiate_dataloader_configuration(self) -> DataLoaderConfiguration:
        params = self.params.dataloader_params

        dataloader_configuration = DataLoaderConfiguration(train_batch_size = self.params.TRAIN_BATCH_SIZE,
                                                           valid_batch_size = self.params.valid_batch_size,
                                                           img_size = self.params.img_size,
                                                           augmentations = self.params.augmentations
                                                           )
        return dataloader_configuration
    

    def initiate_base_model_configuration(self) -> BaseModelConfiguration:
        config = self.config.base_model_config
        params = self.params.base_model_params
        augmentations = self.augmentations
        base_model_configuration = BaseModelConfiguration(root_dir = self.config.root_dir,
                                                           base_model_path = self.config.base_model_path,
                                                           model_name = self.params.MODEL_NAME,
                                                           checkpoint_path = self.params.CHECKPOINT_PATH,
                                                           augmentations = self.augmentations
                                                           )
        return base_model_configuration



    def initiate_model_training_configuration(self) -> ModelTrainingConfiguration:
        config = self.config.model_training_config
        params = self.params.model_training_params

        model_training_configuration = ModelTrainingConfiguration(root_dir = self.config.root_dir,
                                                                  trained_model_path = self.config.root_dir,
                                                                  scheduler = self.params.SCHEDULER,
                                                                  epochs = self.params.EPOCHS,
                                                                  learning_rate = self.params.LEARNING_RATE,
                                                                  min_lr = self.params.MIN_LR,
                                                                  weight_decay = self.params.WEIGHT_DECAY,
                                                                  t_max = self.params.T_MAX,
                                                                  n_accumulate = self.params.N_ACCUMULATE
                                                                  )
        return model_training_configuration










































