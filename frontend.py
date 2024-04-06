import streamlit as st
# from codegeneration import *
from codegeneration2 import *
import pandas as pd


st.set_option('deprecation.showPyplotGlobalUse', False)
col = st.columns(2)
with col[0]:
    st.title("Langchain")
    st.write("Langchain is a tool that allows you to write code in natural language and get the code output.")
data = pd.read_csv("/Users/humphreyhanson/fleet/langchain_rag001/dataset/diamonds.csv")
st.session_state.dataframe = data
  

try:
    # newone = enforce_rules(run_model("what is the price of diamond according to carat and depth?"))
    # st.code(newone)
    newone = code_and_test("visualize the relation between price and depth, and carat of diamond?")
    st.code(newone)
    # timess = exec(newone)
    # st.altair_chart(timess, use_container_width=True)
except AttributeError:
    pass  # Suppress the AttributeError
