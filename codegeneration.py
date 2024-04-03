# scrapping content from a website
from bs4 import BeautifulSoup as soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

# loading files
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd

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

import os
from dotenv import  load_dotenv

load_dotenv()
api_key = os.getenv("open_ai_api")


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


def Document_csv():

    # loading csv files
    csvdocument = pd.read_csv("/Users/humphreyhanson/fleet/langchain_rag001/dataset/penguins.csv")
    return csvdocument

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
    
        # concating all the docs
        # all_docs = "\n\n\n --- \n\n\n".join([doc.page_content for doc in doc_reversed])
        return doc_reversed



def run_model():
    ## Data model
    dataset = Document_loader()
    class code(BaseModel):
        """Code output"""

        prefix: str = Field(description="Description of the visualization in present tense and active voice")
        imports: str = Field(description="Code block import statements")
        code: str = Field(description="Code block not including import statements")

    ## LLM
    model = ChatOpenAI(api_key=api_key, temperature=0, model="gpt-4-turbo-preview")

    # Tool
    code_tool_oai = convert_to_openai_tool(code)

    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[code_tool_oai],
        tool_choice={"type": "function", "function": {"name": "code"}},
    )

    # Parser
    parser_tool = PydanticToolsParser(tools=[code])

    ## Prompt
    template = """You are a coding assistant with expertise in Python, Streamlit and Vega-Altair visualization library. \n 
        Here is a full information and documentation on the dataset and libraries: 
        \n ------- \n
        {context} 
        \n ------- \n
        Answer the user question based on the above provided documentation. \n
        Ensure any code you provide can be executed with all required imports and variables defined. \n
        Code should be executable within streamlit and compatible with streamlit chart elements, specifically st.altair_chart. \n
        Structure your answer with a description of the visualization in present tense and active voice . \n
        Then list the imports. And finally list the functioning code block. \n
        Here is the user question: \n --- --- --- \n {question}"""
    
        # Prompt
    prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question", "generation", "error"],
        )

        # Chain
    chain = (
            {
                "context": lambda x: dataset,
                "question": itemgetter("question"),
            }
            | prompt
            | llm_with_tool
            | parser_tool
        )
    
    outputs = chain.invoke({"question": "create a plot visualizing the relationship between species, boddymass and sex?"})
    print("\n")
    print(outputs[0].prefix)
    print("\n")
    print(outputs[0].imports)
    print(outputs[0].code)
    print("\n")