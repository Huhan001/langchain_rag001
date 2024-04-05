# scrapping content from a website
from bs4 import BeautifulSoup as soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

# loading files
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd


# langchain orcherstration files
from operator import itemgetter
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI


# # langsmith tracing
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable

# .env files
import os
from dotenv import  load_dotenv

# loading enviroment
load_dotenv()
api_key = os.getenv("open_ai_api")

# document loader
def Document_loader():
    
        # loading streamlit chart docs.
        url = "https://docs.streamlit.io/library/api-reference/charts/st.altair_chart"
        loader = RecursiveUrlLoader(url = url, 
                                    max_depth= 20, extractor= lambda x: soup(x, 'html.parser').text)
        streamlit_docs = loader.load()
    
        # loading csv files
        loader_csv = CSVLoader(file_path="/Users/humphreyhanson/fleet/langchain_rag001/dataset/penguins.csv")
        csv_docs = loader_csv.load()

    
        # adding all docs
        streamlit_docs.extend(csv_docs)
        # streamlit_docs.extend(csvdocument)
    
        # sort the list based on url in metadata
        doc_sorted = sorted(streamlit_docs, key = lambda x: x.metadata['source'])
        doc_reversed = list(reversed(doc_sorted))
    

        return doc_reversed