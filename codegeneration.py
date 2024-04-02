# scrapping content from a website
from bs4 import BeautifulSoup as soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

# loading files
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd

# # langsmith tracing
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable


def doc_loader():

    # loading streamlit chart docs.
    url0 = "https://docs.streamlit.io/library/api-reference/charts"
    loader = RecursiveUrlLoader(url = url0, 
                                max_depth= 20, extractor= lambda x: soup(x, 'html.parser').text)
    streamlit_docs = loader.load()

    # loading seaborn docs
    url1 = "https://seaborn.pydata.org/api.html"
    url2 = "https://seaborn.pydata.org/examples/index.html"
    loader1 = RecursiveUrlLoader(url = url1, 
                                max_depth= 20, extractor= lambda x: soup(x, 'html.parser').text)
    seaborn_docs = loader1.load()

    loader2 = RecursiveUrlLoader(url = url2, 
                                max_depth= 20, extractor= lambda x: soup(x, 'html.parser').text)
    seaborn_docs1 = loader2.load()

    # loading csv files
    loader_csv = CSVLoader(file_path="/Users/humphreyhanson/fleet/langchain_rag001/dataset/penguins.csv")
    csv_docs = loader_csv.load()

    # adding all docs
    streamlit_docs.extend(seaborn_docs)
    streamlit_docs.extend(seaborn_docs1)
    streamlit_docs.extend(csv_docs)

    # sort the list based on url in metadata
    doc_sorted = sorted(streamlit_docs, key = lambda x: x.metadata['source'])
    doc_reversed = list(reversed(doc_sorted))

    # concating all the docs
    all_docs = "\n\n\n --- \n\n\n".join([doc.page_content for doc in doc_reversed])
   
    return all_docs

