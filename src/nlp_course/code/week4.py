
# week 3 
# for each language train a classifier 
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import os
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import nest_asyncio
import torch
import instructor
from pydantic import BaseModel
from openai import OpenAI
from torch import nn
from torch.optim import Adam
from torch.nn import BCELoss
from dataclasses import dataclass
import numpy as np
nest_asyncio.apply()

current_dir = os.getcwd()
if current_dir.endswith("code"):
    os.chdir("..")
else:
    print("current dir", current_dir)

ds_train = pd.read_parquet("dataset/train_df.parquet")
ds_val = pd.read_parquet("dataset/val_df.parquet")

filtered_df_train = ds_train[ds_train["answer"] != "no"]
filtered_df_val = ds_val[ds_val["answer"] != "no"]

from pydantic import BaseModel, Field, ValidationInfo, field_validator

class SpanPrediction(BaseModel):
    answer_context: str = Field(..., description="The context of the answer")

    """ @field_validator('answer_context')
    @classmethod
    def validate_answer(cls, value: str, info: ValidationInfo) -> str:
        print("info", info)
        context = info.context
        print("context", context)
        if value not in context:
            raise ValueError(f"answer_context must be present in the provided context: {context}")
        return value """
    
from typing import List
    
def predict_span(
    context: List[str],
    question: List[str],
    answer: List[str]
) -> List[SpanPrediction]:
    assert len(context) == len(question) == len(answer)
    
    def format_prompt(
        context: str,
        question: str,
        answer: str
    ) -> List[dict]:
        return [
            {
                "role": "system",
                "content": f"""you are a span labeler. Given a question and a correct answer, 
                                identify the span in the context that answers the question.
                                question: {question}
                                answer: {answer}
                            """
            },
            {
                "role": "user",
                "content": f"""
                            context: {context}
                            """
            }
        ]
    
    prompts = [format_prompt(c, q, a) for c, q, a in zip(context, question, answer)]
    #flatten the list of prompts
    flattened_prompts = [item for sublist in prompts for item in sublist]
    print("flattened_prompts", flattened_prompts)
    client = instructor.from_openai(OpenAI())
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=flattened_prompts, #flatten the list of prompts
        response_model=List[SpanPrediction],
        validation_context=context
    )

    return response

filtered_df_train[:1]
predict_span(filtered_df_train["context"].tolist(), filtered_df_train["question"].tolist(), filtered_df_train["answer"].tolist())
    
