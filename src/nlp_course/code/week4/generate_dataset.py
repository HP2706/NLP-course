import pandas as pd
import os
from datasets import Dataset
from transformers import AutoTokenizer

current_dir = os.getcwd()
if current_dir.endswith("code"):
    os.chdir("..")
else:
    print("current dir", current_dir)

ds_train = pd.read_parquet("dataset/train_df.parquet")
ds_val = pd.read_parquet("dataset/val_df.parquet")
