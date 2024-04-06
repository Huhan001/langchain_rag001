from legacy_research.freshStart import  *
from legacy_research.GraphGeneratingLLM import *
from legacy_research.SelfCorectingLLM import *
from LibrariesUndDataLoad import *

if __name__ == "__main__":
    path = "/Users/humphreyhanson/fleet/langchain_rag001/dataset/penguins.csv"
    newone = replace_loading_dataset_with_csv_read(run_model(), path)
    exec(newone)
    # finalise("show me the species accoding to bodymass and sex?")
