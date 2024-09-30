import numpy as np
from typing import Literal, Optional
from common import app, SYSTEM_MESSAGE
from inference import Inference
import pandas as pd
from transformers import AutoTokenizer
from utils import construct_prompt
import os
from utils import exact_match, f1_score

current_dir = os.getcwd()
print(current_dir)
if current_dir.endswith("code"):
    os.chdir("..")
    print(os.getcwd())
else:
    print("current dir", current_dir)

def print_and_eval(
    df: pd.DataFrame, 
    guesses : list[str],
    model_name : str,
    answer_key : Literal['answer', 'answer_inlang'] = 'answer'
):
    expected = df[answer_key].tolist()
    assert len(expected) == len(guesses)
    
    np_guesses = np.array(guesses)
    np_labels = np.array(expected)

    for lang in ['fi', 'ja', 'ru']:
        print(f"evaluating {lang}")
        lang_idxs = df['lang'] == lang
        guesses_lang = np_guesses[lang_idxs]
        labels_lang = np_labels[lang_idxs]

        print("guesses_lang", guesses_lang.shape)
        print("labels_lang", labels_lang.shape)
        if len(guesses_lang) == 0: 
            continue
        print("exact_match", exact_match(guesses_lang, labels_lang))
        print("f1_score", f1_score(guesses_lang, labels_lang))
        
    print("total exact_match", exact_match(np_guesses, np_labels))
    print("total f1_score", f1_score(np_guesses, np_labels))
    
    
    #examine performance on unanswerable questions
    unanswerable_df = df[df['answer'] == 'no']
    print("unanswerable_df", unanswerable_df.shape)
    unanswerable_guesses = np_guesses[unanswerable_df.index]
    print("unanswerable_guesses", unanswerable_guesses.shape)
    print("unanswerable_labels", np_labels[unanswerable_df.index].shape)
    em, em_idxs = exact_match(unanswerable_guesses, np_labels[unanswerable_df.index])
    f1, f1_idxs = f1_score(unanswerable_guesses, np_labels[unanswerable_df.index])
    print("exact_match", em)
    print("f1_score", f1)
    
    df = pd.DataFrame({'answer': np_labels, 'guess': np_guesses})
    df.to_csv(f'eval_results_{model_name}.csv', index=False)
    
    

def eval(
    inlang: bool = True, 
    no_finetune: bool = False,
    model_name : str = 'axo-2024-09-24-15-06-12-09cf'
):
    if no_finetune:
        print('using original llama-3.1-8b-instruct')
        model_name = 'meta_llama_no_finetune'
    else:
        print('using finetuned model', model_name)
    inference = Inference(run_name=model_name, no_finetune=no_finetune)
    ds_val = pd.read_parquet('dataset/val_df.parquet')
    
    if inlang:
        ds_val = ds_val[ds_val['answer_inlang'].notna()]
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    print("ds_val.columns", ds_val.columns)
    
    prompts = [construct_prompt(tokenizer, ds_val.iloc[i]['question'], ds_val.iloc[i]['context'], None) for i in range(len(ds_val))]
    guesses = inference.batched_inference.remote(prompts)
    print_and_eval(
        ds_val, 
        guesses, 
        model_name=model_name,
        answer_key='answer_inlang' if inlang else 'answer'
    )
        
@app.local_entrypoint()
def eval_model():
    eval(inlang=False, no_finetune=True)
    eval(inlang=False, no_finetune=False, model_name='axo-2024-09-24-20-47-06-872a')