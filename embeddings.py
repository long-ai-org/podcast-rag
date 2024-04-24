import os

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores.zilliz import Zilliz
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

embeddings = AzureOpenAIEmbeddings(model=os.environ.get("OPENAI_EMBEDDINGS_MODEL"))
connection_args = {
    "uri": os.environ.get("ZILLIZ_CLOUD_URI"),
    "token": os.environ.get("ZILLIZ_CLOUD_API_KEY"),
}

COLLECTION_NAME = "podcast_fragen_antworten_chunks"

df = pd.read_excel("input/Podcast Chunks.xlsx")

for index, row in df.iterrows():
    Zilliz(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name=COLLECTION_NAME,
    ).from_texts(
        texts=[row["Chunk"]],
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args=connection_args,
        auto_id=True,
    )
