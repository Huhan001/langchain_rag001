from Application.LibrariesUndDataLoad import *
import re



def run_model():
    ## Data model
    dataset = loading_dataset()
    class code(BaseModel):
        """Code output"""

        prefix: str = Field(description="Description of the visualization explains the plot created. help with understanding the plot")
        imports: str = Field(description="Code block import statements")
        code: str = Field(description="Code block not including import statements")

    ## LLM
    model = ChatOpenAI(api_key=api_key, temperature=0, model="gpt-3.5-turbo-0125")

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
    template = """You are a coding assistant with expertise in Python, Vega-Altair visualization library. \n 
        Here is a full information and documentation on the dataset and libraries: 
        \n ------- \n
        {context} 
        \n ------- \n
        Answer the user question based on the above documentation. \n
        Ensure any code you provide can be executed with all required imports and variables defined. \n
        make sure that the deta loaded in the code is always equal to dataset = loading_dataset()  \n
        dataset = loading_dataset() \n
        Please make sure and always have the deta loaded in the code is always equal to dataset = loading_dataset() format\n
        Structure your answer with a description explaining the visualization created. explain so the user understand the visualization . \n
        Again please, please make sure that the deta loaded in the code is always equal to dataset = loading_dataset() \n
        conclude the code with st.altair_chart(chart, use_container_width=True) \n
        "st.altair_chart(chart, use_container_width=True)" very important to execute the generated code and display \n
        make sure chart at the end is set to st.altair_chart(c, use_container_width=True) to display the chart \n
        Then list the imports. And finally list the functioning code block. \n
        end with st.altair_chart(chart, use_container_width=True) \n
        Here is the user question: \n --- --- --- \n {question}"""
    
        # Prompt
    prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
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
    
    outputs = chain.invoke({"question": "what is the relationship between species, boddymass and sex?, use barxhart to show the relationship"})
    
    full_code = outputs[0].imports + "\n" + outputs[0].code
    return full_code


def replace_loading_dataset_with_csv_read(code):
    # Replace occurrences of "loading_dataset()" with "pd.read_csv(csv_path)"
    modified_code = code.replace("loading_dataset()", "st.session_state.dataframe.copy()")
    print(modified_code)
    return modified_code


def check_chart_location(code):
    # Regular expression pattern to match 'chart' surrounded by non-alphanumeric characters
    pattern = r'(?<!\w)chart(?!\w)'
    
    # Search for the pattern in the code
    match = re.search(pattern, code)
    
    # If 'chart' is found and is not part of another variable or function name
    if match:
        return True
    
    return False

def check_data_input(code):
    if 'dataset = st.session_state.dataframe.copy()' not in code:
        return False
    return True

def clean_code(code):
    # Add logic to clean the code if necessary
    cleaned_code = code.strip()
    return cleaned_code

def enforce_rules(code):
    # Replace loading_dataset() with st.session_state.dataframe.copy() first
    inside_code = replace_loading_dataset_with_csv_read(code)
    
    if not check_chart_location(inside_code):
        # Replace standalone 'chart' with 'st.altair_chart(chart, use_container_width=True)'
        inside_code = re.sub(r'(?<!\w)chart(?!\w)', 'st.altair_chart(chart, use_container_width=True)', inside_code)
    
    # Check data input and insert if necessary
    if not check_data_input(inside_code):
        inside_code = 'dataset = st.session_state.dataframe.copy()\n' + inside_code
    
    # Clean the code
    code_cleaned = clean_code(inside_code)
    
    return code_cleaned



