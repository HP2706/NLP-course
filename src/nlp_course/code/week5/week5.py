from transformers import AutoTokenizer
import pandas as pd
from typing import Optional
import os

current_dir = os.getcwd()
print(current_dir)
if current_dir.endswith("code"):
    os.chdir("..")
    print(os.getcwd())
else:
    print("current dir", current_dir)

def construct_prompt(
    tokenizer: AutoTokenizer,
    question : str,
    context : str,
    answer : Optional[str] = None,
    tokenize : bool = False
):
    messages = [
        {"role": "system", "content": "Given the users context and question, answer the question."},
        {"role": "user", "content": f"Question: {question}\nContext: {context}"}
    ]
    if answer:
        messages.append({"role": "assistant", "content": f"{answer}"})
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=False)
    return prompt

def construct_input_output_from_df(
    df: pd.DataFrame, 
    tokenizer: AutoTokenizer
) -> pd.DataFrame:
    df['prompt_str'] = df.apply(
        lambda x: construct_prompt(
            tokenizer, x['question'], x['context'], x['answer'], tokenize=True
    ), axis=1)

    