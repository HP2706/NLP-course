from typing import Optional
from common import app, SYSTEM_MESSAGE
from inference import Inference
import pandas as pd
from transformers import AutoTokenizer
from utils import construct_prompt
import os

current_dir = os.getcwd()
print(current_dir)
if current_dir.endswith("code"):
    os.chdir("..")
    print(os.getcwd())
else:
    print("current dir", current_dir)

def _eval_model():
    inference = Inference('axo-2024-09-24-15-06-12-09cf')
    print("os current dir", os.getcwd())
    print("files in dir", os.listdir())
    print("exists", os.path.exists('dataset/val_df.parquet'))
    
    
    ds_val = pd.read_parquet('dataset/val_df.parquet')
    ds_val_inlang = ds_val[ds_val['answer_inlang'].notna()][:10]
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    print("ds_val_inlang.columns", ds_val_inlang.columns)
    
    prompts = [construct_prompt(tokenizer, ds_val_inlang.iloc[i]['question'], ds_val_inlang.iloc[i]['context'], None) for i in range(len(ds_val_inlang))]
    print("prompts", prompts)
    guesses = [inference.non_streaming.remote(prompts[i]) for i in range(len(prompts))]
    
    for i, guess in enumerate(guesses):
        #print(f"input: {prompts[i]}")
        print(f"should be output: {ds_val_inlang.iloc[i]['answer_inlang']}")
        print(f"model output: {guess}")
        print("\n\n")
        
@app.local_entrypoint()
def eval_model():
    _eval_model()
    
    
#_eval_model()