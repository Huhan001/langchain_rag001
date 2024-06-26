import streamlit as st
from SelfCorrectiveLLM import *
import pandas as pd


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

columns = st.columns([3,6,3])
with columns[1]:
    data = pd.read_csv("/Users/humphreyhanson/fleet/langchain_rag001/dataset/diamonds.csv")
    st.session_state.dataframe = data

    question = st.text_area(placeholder="Your inquiry goes inhere", value="", label="Write out your question", height=1)
    Vizbutton = st.button("Generate Plot",type="secondary")

    if Vizbutton:
        try:
            if question:
                code = code_and_test(question)
                st.write(Codeinterpreter_explainer(code))
        except AttributeError:
            pass
