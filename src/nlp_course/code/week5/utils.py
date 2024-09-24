from typing import Optional
from transformers import AutoTokenizer


remove_sentence = """Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024"""


SYSTEM_MESSAGE = """
Given the users context(in english) and question(in either finnish, russian or japanese), answer the question(which is a substring of the context) in the same language as the question.
If the question is not answerable, return "None".

EXAMPLES:
'<|begin_of_text|><|start_header_id|>system<|end_header_id|>
\nCutting Knowledge Date: December 2023
Today Date: 26 Jul 2024
Given the users context(in english) and question(in either finnish, russian or japanese), answer the question(which is a substring of the context) in the same language as the question.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuestion: What is the capital of France?
Context: France is a country in Europe.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nParis<|eot_id|>'
"""

def construct_prompt(
    tokenizer : AutoTokenizer,
    question : str,
    context : str,
    answer : Optional[str] = None,
    tokenize : bool = False
) -> str:
    
    messages = [
        {"role": "system", "content": f"{SYSTEM_MESSAGE}"},
        {"role": "user", "content": f"Question: {question}\nContext: {context}"},
    ]
    if answer:
        messages.append({"role": "assistant", "content": f"{answer}"})

    prompt = tokenizer.apply_chat_template(
        conversation=messages, 
        tokenize=tokenize, 
        add_generation_prompt=False, 
        format="chatml"
    )

    return prompt