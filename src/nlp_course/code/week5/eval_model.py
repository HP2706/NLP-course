from common import app, SYSTEM_MESSAGE
from inference import Inference
import pandas as pd
from transformers import AutoTokenizer
import os

current_dir = os.getcwd()
print(current_dir)
if current_dir.endswith("week5"):
    os.chdir("../..")
    print(os.getcwd())
else:
    print("current dir", current_dir)

def _eval_model():
    inference = Inference('axo-2024-09-24-09-41-09-6306')
    print("exists", os.path.exists('week5/data/finetune_alpaca_inlang_val.jsonl'))
    df = pd.read_json('week5/data/finetune_alpaca_inlang_val.jsonl', lines=True)[2:10]
    print("df", len(df), df.columns)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    def format_prompt(input, instruction):
        return f"""[INST] Given the user question(in finnish, russian or japanese), answer the question in the same language as the question(finnish, russian or japanese)
        {input}
        {instruction} [/INST]"""
    
    alpaca_strings = [
        format_prompt(input, instruction) for input, instruction in zip(df["question"], df["context"])
    ]
    
    #print("alpaca_strings", alpaca_strings)
    guesses = [inference.non_streaming.remote(alpaca_strings) for alpaca_strings in alpaca_strings]
    
    for i, row in range(len()):
        print(f"input: {alpaca_strings[i]}")
        print(f"should be output: {row['answer_inlang']}")
        print(f"model output: {guesses[i]}")
        print("\n\n")
        
@app.local_entrypoint()
def eval_model():
    _eval_model()