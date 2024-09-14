# various utils for distilling features 
from openai import OpenAI
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import Awaitable
import instructor
import asyncio
from typing import AsyncGenerator
import nest_asyncio
from openai import AsyncOpenAI
from datasets import load_dataset
import pandas as pd
import os
from tqdm.asyncio import tqdm as async_tqdm
from typing import List

ds = load_dataset("coastalcph/tydi_xor_rc")
ds_val = ds["validation"].to_pandas()
ds_train = ds["train"].to_pandas()
ds_train = ds_train[ds_train['lang'].isin(['fi', 'ja', 'ru'])]
ds_val = ds_val[ds_val['lang'].isin(['fi', 'ja', 'ru'])]

class TranslationResponse(BaseModel):
    original_text : str = Field(description="The original text that was translated")
    translated_text : str = Field(description="The translated text into English from either Finnish, Japanese or Russian")
    

from typing import Optional
async def translate_chunk(
    chunk: List[tuple[str, str]]
) -> Optional[List[TranslationResponse]]:
    client = instructor.from_openai(AsyncOpenAI())

    text_chunks = '\n'.join([f'lang: {t[0]} \n text: {t[1]}' for t in chunk])
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=List[TranslationResponse],
        messages=[
            {
                "role": "system",
                "content": "You are a translation assistant that translates text from Finnish, Japanese or Russian to English. Please output a list of TranslationResponse objects ordered by the original input list."
            },
            {
                "role": "user",
                "content": f"Translate the following text to English: {text_chunks}"
            }
        ]
    )
    if len(response) != len(chunk):
        print(f"Response length ({len(response)}) does not match chunk length ({len(chunk)})")
        return None

    return response

async def translate_questions(df: pd.DataFrame) -> List[TranslationResponse]:
    batch_size = 2

    if df.empty:
        print("No rows to translate.")
        return df
    
    # Initialize the question_translated column for all rows
    df["question_translated"] = ""
    
    async def process_batch(task_id : int, batch) -> Awaitable[List[TranslationResponse]]:
        translations = await translate_chunk(
            [(row["lang"], row["question"]) for _, row in batch.iterrows()]
        )
        return task_id, translations
    
    
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    all_translations = []
    
    tasks = [process_batch(task_id=i, batch=batch) for i, batch in enumerate(batches)]
    
    async for translation in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Translating"):
        all_translations.append(await translation)

    all_translations.sort(key=lambda x: x[0])
    
    flattened_translations = [item for sublist in all_translations for item in sublist[1]]
    # Update only the rows that needed translation
    
    
    for i, translation in enumerate(flattened_translations):
        if translation is not None:
            df.loc[df.index[i], "question_translated"] = translation.translated_text
        else:
            print(f"Translation for row {i} is None")
    return df

""" 
# Wrap the translation process in a try-except block
try:
    translated_df = asyncio.run(translate_questions(ds_val))
    current_dir = os.getcwd()
    if current_dir.endswith("week1"):
        os.chdir("..")
    else:
        print("current dir", current_dir)
        
    path = "nlp_course/dataset"
    os.makedirs(path, exist_ok=True)
    translated_df.to_parquet(f"{path}/translated_df_val.parquet", index=False)
    os.chdir(current_dir)
    print("Translation completed successfully.")
except Exception as e:
    print(f"An error occurred during translation: {str(e)}")
 """

from nltk import ngrams
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import ngrams
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

import os

os.chdir("..")
print(os.getcwd())
df_train = pd.read_parquet("dataset/train_df.parquet")
df_val = pd.read_parquet("dataset/val_df.parquet")

print("columns", df_train.columns)

from nltk.lm import Lidstone  # Add this import at the top of your file
from typing import Literal, Optional

class NGramTrainer:
    model: Optional[MLE] = None
    def __init__(
        self, 
        ds_train: pd.DataFrame,
        ds_val: pd.DataFrame,
        n: int, 
        lang: Optional[Literal['ja', 'ru', 'fi']] = None, 
        on_context: bool = False
    ):

        
        if lang:
            ds_train = ds_train[ds_train['lang'] == lang]
            ds_val = ds_val[ds_val['lang'] == lang]
            print("ds_train", ds_train.head())
            print("ds_val", ds_val.head())
            print("columns", ds_train.columns)
            self.ds_train = ds_train['question_tokens'].tolist()
            self.ds_val = ds_val['question_tokens'].tolist()
        elif on_context:
            self.ds_train = ds_train['context_tokens'].tolist()
            self.ds_val = ds_val['context_tokens'].tolist()
        else:
            raise ValueError("lang must be provided if on_context is False")
        
        self.n = n
        self.lang = lang
        self.on_context = on_context
        self.model = None
        
    def fit(self):

        texts = self.ds_train
        train_data, padded_sents = padded_everygram_pipeline(self.n, texts)
        
        # Create and train the model
        model = Lidstone(gamma=0.1)  # gamma is the smoothing parameter
        model.fit(train_data, padded_sents)
        self.model = model
        return model

    def evaluate(self):
        '''evaluate on the validation set'''
        if self.model is None:
            raise ValueError("Model not trained yet")
        texts = self.ds_val
        _, padded_sents = padded_everygram_pipeline(self.n, texts)
        perplexity = self.model.perplexity(padded_sents)
        return perplexity

Trainer = NGramTrainer(ds_train, ds_val, 2, lang='ja')
Trainer.fit()
Trainer.evaluate()


#evaluate_model(model, ds_val)
# Now you can use the model for various tasks, e.g., calculating probabilities
