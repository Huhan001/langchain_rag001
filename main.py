from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# prompt template guide
from langchain_core.prompts import ChatPromptTemplate

# stringParser
from langchain_core.output_parsers import StrOutputParser

#retreiver chain
#- retrievers give access to knowledge by picking the most relevant documents from a large collection of documents
from langchain_community.document_loaders import WebBaseLoader

#embedings ranked together.
from langchain_openai import OpenAIEmbeddings

# building out index
from langchain_community.vectorstores import FAISS 
from langchain.text_splitter import RecursiveCharacterTextSplitter

#rettiver chain
from langchain.chains.combine_documents import create_stuff_documents_chain



def RagApplication():

    # load enviroment .env file
    load_dotenv()
    openai_key = os.getenv('OPEN_AI_API')
    openai_chat = ChatOpenAI(openai_api_key= openai_key)
    
    # chat prompt
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "you are world class technical documentation writer"),
    #      ("user", "{input}")
    # ])

    # chain = prompt | openai_chat
    # print(chain.invoke({"input": "how can langsmith help with testing?"}))
    
    # convertint to a string instead of a sentence as it is right now
    Output_Parser = StrOutputParser()

    chain = prompt | openai_chat | Output_Parser
    print(chain.invoke({"input": "how can langsmith help with testing?"}))

    #retreiver by passing in web and scrapping with beautiful soap
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()

    #make sure you have the embedings packages installed
    embeddings = OpenAIEmbeddings()

    #build index
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    #lets set up a chain that takes a question and retrieves documents then generaets an answer.
    prompt = ChatPromptTemplate.















if __name__ == "__main__":
    RagApplication()