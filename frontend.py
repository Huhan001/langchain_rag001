import streamlit as st
from SelfCorrectiveLLM import *
import pandas as pd


st.set_option('deprecation.showPyplotGlobalUse', False)
col = st.columns([0.8, 0.2])

with col[0]:
    data = pd.read_csv("/Users/humphreyhanson/fleet/langchain_rag001/dataset/diamonds.csv")
    st.session_state.dataframe = data

    question = st.text_area(placeholder="Your inquiry goes inhere", value="", label="Write out your question")
    Vizbutton = st.button("Generate Plot",type="secondary")

    if Vizbutton:
        try:
            if question:
                code = code_and_test(question)
                st.write(Codeinterpreter_explainer(code))
        except AttributeError:
            pass
