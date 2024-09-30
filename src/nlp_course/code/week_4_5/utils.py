from typing import Optional
from transformers import AutoTokenizer
import numpy as np

import string
def normalize_text(text):
    # Fix whitespaces, convert lowercase
    text = " ".join(text.split()).lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text

def exact_match(preds : list[str], labels : list[str]) -> tuple[float, np.ndarray]:
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
        
    preds = np.vectorize(normalize_text)(preds)
    labels = np.vectorize(normalize_text)(labels)

    idxs = preds == labels
    return np.mean(idxs), idxs

#inspired by: https://github.com/terru3/t5-qa/blob/bf853e859506a7acc4bb0043fc719c292bd155b2/metrics.py
def f1_score(preds, labels) -> tuple[float, np.ndarray]:
    preds = np.vectorize(normalize_text)(preds)
    labels = np.vectorize(normalize_text)(labels)

    f1_list = []

    # Calculates F-1 Score for each pair of preds & labels
    for i in range(len(preds)):
        pred_tokens = preds[i].split()
        act_tokens = labels[i].split()

        common_tokens = set(pred_tokens) & set(act_tokens)
        if len(common_tokens) == 0:
            f1_list.append(0)
        else:
            pre = len(common_tokens) / len(pred_tokens)
            rec = len(common_tokens) / len(act_tokens)
            f1 = 2 * (pre * rec) / (pre + rec)
            f1_list.append(f1)

    return np.mean(f1_list), np.array(f1_list)



remove_sentence = """Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024"""

def gen_system_message(message: str):
    return f"""
    The following is a qa task, given the context and question, answer the question.The answer will be in the context.
    {message}
    If the question is not answerable, output: "None".
    """

INLANG_SYSTEM_MESSAGE = gen_system_message('You should answer in the same language as the question.')
EN_SYSTEM_MESSAGE = gen_system_message('You should answer in English.')


def construct_prompt(
    tokenizer : AutoTokenizer,
    question : str,
    context : str,
    answer : Optional[str] = None,
    tokenize : bool = False,
    is_inlang : bool = False
) -> str:
    
    messages = [
        {"role": "system", "content": f"{INLANG_SYSTEM_MESSAGE if is_inlang else EN_SYSTEM_MESSAGE}"},
        {"role": "system", "content": f"Context: {context}"},
        {"role": "user", "content": f"Question: {question}"},
    ]
    if answer:
        messages.append({"role": "assistant", "content": f"{answer}"})

    prompt = tokenizer.apply_chat_template(
        conversation=messages, 
        tokenize=tokenize, 
        add_generation_prompt=False, 
        format="chatml"
    )
    
    if not answer:
        added = f'<|start_header_id|>assistant<|end_header_id|>'
        prompt = prompt + added
        
    return prompt