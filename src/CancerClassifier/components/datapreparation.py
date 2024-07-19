import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from CancerClassifier import logger
from CancerClassifier.config import DataConfiguration




class DataPreparation():
    def __init__(self, setup: DataConfiguration):
        self.setup = setup
        


    def get_train_file_path(self, image_id):
        return f"{self.setup.train_images_path}/{image_id}.jpg"



    def get_transformed_data(self):
        try:
            train_image_paths = sorted(glob.glob(f"{self.setup.train_images_path}/*.jpg"))
            train_image_paths = [path.replace("\\", "/") for path in train_image_paths]

            df = pd.read_csv(self.setup.train_csv_path)

            print("        df.shape, # of positive cases, # of patients")
            print("original>", df.shape, df.target.sum(), df["patient_id"].unique().shape)

            df_positive = df[df["target"] == 1].reset_index(drop=True)

            print(f"Number of Positive Cases: {df_positive.shape[0]}")

            df_negative = df[df["target"] == 0].reset_index(drop=True)

            print(f"Number of Negative Cases: {df_negative.shape[0]}")

            df = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0]*20, :]])  # positive:negative = 1:20

            print("filtered>", df.shape, df.target.sum(), df["patient_id"].unique().shape)

            df['file_path'] = df['isic_id'].apply(self.get_train_file_path)
            df = df[ df["file_path"].isin(train_image_paths) ].reset_index(drop=True)
            df

            sgkf = StratifiedGroupKFold(n_splits=self.setup.n_folds)
            for fold, ( _, val_) in enumerate(sgkf.split(df, df.target, df.patient_id)):
                df.loc[val_ , "kfold"] = int(fold)

            df.to_csv(f"{self.setup.root_dir}/transformed-train-data.csv")
            logger.info("Transformed CSV Data Saved.")
        except Exception as e:
            print(e)