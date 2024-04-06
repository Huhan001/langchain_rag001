from dependancies import *
import re
import streamlit as st



def run_model(user_question):
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
        Then list the imports. And finally list the functioning code block. \n
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
    
    outputs = chain.invoke({"question": user_question})
    
    full_code = outputs[0].imports + "\n" + outputs[0].code
    return full_code


def replace_loading_dataset_with_csv_read(code):
    # Replace occurrences of "loading_dataset()" with "pd.read_csv(csv_path)"
    modified_code = code.replace("loading_dataset()", "st.session_state.dataframe.copy()")
    print(modified_code)
    return modified_code



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
    
    # Check data input and insert if necessary
    if not check_data_input(inside_code):
        inside_code = 'dataset = st.session_state.dataframe.copy()\n' + inside_code
    
    # Extract the last variable name
    last_variable_match = re.findall(r'\b(\w+)\s*$', inside_code)
    last_variable = last_variable_match[0] if last_variable_match else None
    
    if last_variable:
        # Inject the last variable into st.altair_chart syntax
        inside_code += f'\nst.altair_chart({last_variable}, use_container_width=True)'

    # Clean the code
    code_cleaned = clean_code(inside_code)
    
    return code_cleaned



def code_and_test(user_question):
    while True:
        try:
            generated_code = run_model(user_question)
            enforced_code = enforce_rules(generated_code)
            exec(enforced_code, globals())
            print("Code execution successful.")
            return enforced_code
        except AttributeError:
            # Ignore AttributeError related to session state initialization failure
            print("Ignoring AttributeError related to session state initialization failure.")
            print("Stopping code generation and testing...")
        except Exception as e:
            print("Error occurred during code execution:", e)
            print("Rerunning code generation and testing...")
            continue
        break  # Stop the loop after successful code execution

