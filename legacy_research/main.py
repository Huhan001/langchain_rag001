from legacy_research.freshStart import  *
from legacy_research.GraphGeneratingLLM import *
from legacy_research.SelfCorectingLLM import *
from LibrariesUndDataLoad import *

if __name__ == "__main__":
    path = "/Users/humphreyhanson/fleet/langchain_rag001/dataset/penguins.csv"
    newone = replace_loading_dataset_with_csv_read(run_model(), path)
    exec(newone)
    # finalise("show me the species accoding to bodymass and sex?")

try:
    # newone = enforce_rules(run_model("what is the price of diamond according to carat and depth?"))
    # st.code(newone)
    newone = code_and_test("visualize the relation between price and depth, and carat of diamond?")
    st.code(newone)
    # timess = exec(newone)
    # st.altair_chart(timess, use_container_width=True)
except AttributeError:
    pass  # Suppress the AttributeError