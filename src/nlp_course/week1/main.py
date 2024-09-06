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
        model="gpt-4-0125-preview",
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



