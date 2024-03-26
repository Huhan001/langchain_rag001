import os
from dotenv import  load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
# this converts the message into a string
# the message we get from the chatbot is a dictionary


load_dotenv()
    
    # set api key
api_key = os.getenv("open_ai_api")

def simple_llm(input):
    
    llm = ChatOpenAI(openai_api_key = api_key)
    output = llm.invoke(input)
    print(output)
    

def Guided_response(provided):

    llm = ChatOpenAI(openai_api_key = api_key)

    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class developer in langchain."),
    ("user", "{input}")])

    output_parser = StrOutputParser()
    # convert to string

    chain = prompt | llm | output_parser
    output = chain.invoke({"input": provided})
    print(output)


def web_based_loader():
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader("https://people.sc.fsu.edu/~jburkardt/data/csv/oscar_age_female.csv")
    docs = loader.load()

    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key = api_key)

    from langchain_community.vectorstores import FAISS 
    vector = FAISS.from_documents(docs, embeddings)

    from langchain.chains.combine_documents import create_stuff_documents_chain

    llm = ChatOpenAI(openai_api_key = api_key)

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    from langchain.chains import create_retrieval_chain

    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": "When was the movie dangeros made? and who is the actor?"})
    print(response["answer"])
   