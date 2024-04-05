from typing import Dict, TypedDict
from dependancies import *


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


def generate(state: GraphState):

    # loading data and context
    context_data = Document_loader()

    """
    Generate a code solution based on streamlit, csv data and the input question
    with optional feedback from code execution tests

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    ## State
    state_dict = state["keys"]
    question = state_dict["question"]
    iter = state_dict["iterations"]

    ## Data model
    class code(BaseModel):
        """Code output"""

        prefix: str = Field(description="Description of the visualization in present tense and active voice")
        imports: str = Field(description="Code block import statements")
        code: str = Field(description="Code block not including import statements")

    ## LLM
    model = ChatOpenAI(api_key= api_key, temperature=0, model="gpt-4-turbo-preview", streaming=True)

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

    ## Generation
    if "error" in state_dict:
        print("---RE-GENERATE SOLUTION w/ ERROR FEEDBACK---")

        error = state_dict["error"]
        code_solution = state_dict["generation"]

        # Udpate prompt
        addendum = """  \n --- --- --- \n You previously tried to solve this problem. \n Here is your solution:  
                    \n --- --- --- \n {generation}  \n --- --- --- \n  Here is the resulting error from code 
                    execution:  \n --- --- --- \n {error}  \n --- --- --- \n Please re-try to answer this. 
                    Structure your answer description of the visualization in present tense and active voice. 
                    \n Then list the imports. And finally list the functioning code block. 
                    \n Here is the user question: \n --- --- --- \n {question}"""
        
        template = template + addendum

        # Prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question", "generation", "error"],
        )

        # Chain
        chain = (
            {
                "context": lambda _: context_data,
                "question": itemgetter("question"),
                "generation": itemgetter("generation"),
                "error": itemgetter("error"),
            }
            | prompt
            | llm_with_tool
            | parser_tool
        )

        code_solution = chain.invoke(
            {"question": question, "generation": str(code_solution[0]), "error": error}
        )

    else:
        print("---GENERATE SOLUTION---")

        # Prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        # Chain
        chain = (
            {
                "context": lambda _: context_data,
                "question": itemgetter("question"),
            }
            | prompt
            | llm_with_tool
            | parser_tool
        )

        code_solution = chain.invoke({"question": question})

    iter = iter + 1

    return {
        "keys": {"generation": code_solution, "question": question, "iterations": iter}
    }


def check_code_imports(state: GraphState):
    """
    Check imports

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    ## State
    print("---CHECKING CODE IMPORTS---")
    state_dict = state["keys"]
    question = state_dict["question"]
    code_solution = state_dict["generation"]
    imports = code_solution[0].imports
    iter = state_dict["iterations"]

    try:
        # Attempt to execute the imports
        exec(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        # Catch any error during execution (e.g., ImportError, SyntaxError)
        error = f"Execution error: {e}"
        if "error" in state_dict:
            error_prev_runs = state_dict["error"]
            error = error_prev_runs + "\n --- Most recent run error --- \n" + error
    else:
        print("---CODE IMPORT CHECK: SUCCESS---")
        # No errors occurred
        error = "None"

    return {
        "keys": {
            "generation": code_solution,
            "question": question,
            "error": error,
            "iterations": iter,
        }
    }

def check_code_execution(state: GraphState):
    """
    Check code block execution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    ## State
    print("---CHECKING CODE EXECUTION---")
    state_dict = state["keys"]
    question = state_dict["question"]
    code_solution = state_dict["generation"]
    prefix = code_solution[0].prefix
    imports = code_solution[0].imports
    code = code_solution[0].code
    code_block = imports + "\n" + code
    iter = state_dict["iterations"]

    try:
        # Attempt to execute the code block
        exec(code_block)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        # Catch any error during execution (e.g., ImportError, SyntaxError)
        error = f"Execution error: {e}"
        if "error" in state_dict:
            error_prev_runs = state_dict["error"]
            error = error_prev_runs + "\n --- Most recent run error --- \n" + error
    else:
        print("---CODE BLOCK CHECK: SUCCESS---")
        # No errors occurred
        error = "None"

    return {
        "keys": {
            "generation": code_solution,
            "question": question,
            "error": error,
            "prefix": prefix,
            "imports": imports,
            "iterations": iter,
            "code": code,
        }
    }