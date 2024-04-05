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
    model = ChatOpenAI(api_key= api_key, temperature=0, model="gpt-3.5-turbo", streaming=True)

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
    template = """You are a coding assistant with expertise in Python,Streamlitand Vega-Altair visualization library. \n 
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





def decide_to_finish(state: GraphState):
    """
    Determines whether to continue or finish based on the presence of errors.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO CONTINUE OR FINISH---")
    state_dict = state["keys"]
    error = state_dict["error"]
    iter = state_dict["iterations"]

    if error is None:
        # If there's no error, continue generating solutions
        print("---DECISION: CONTINUE GENERATING SOLUTIONS---")
        return "generate"
    else:
        # If there's an error, but the maximum number of iterations (3) is not reached,
        # retry checking code imports
        print("---DECISION: RE-TRY CHECKING CODE IMPORTS---")
        return "check_code_imports" if iter < 3 else END



def check_code_imports(state: GraphState):
    """
    Check imports and update requirements.txt if necessary.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with potential error and modified requirements.txt
    """

    ## State
    print("---CHECKING CODE IMPORTS---")
    state_dict = state["keys"]
    question = state_dict["question"]
    code_solution = state_dict["generation"]
    imports = code_solution[0].imports

    # Get existing libraries from requirements.txt
    with open("requirements.txt", "r") as f:
        existing_libraries = set(line.strip() for line in f.readlines())

    # Extract libraries from import statements
    imported_libraries = set(import_line.split()[1] for import_line in imports.split("\n") if import_line.startswith("import"))

    # Libraries to be added to requirements.txt
    new_libraries = imported_libraries - existing_libraries

    if new_libraries:
        print("---MISSING LIBRARIES FOUND---")
        print("Libraries to be added to requirements.txt:", new_libraries)

        # Add new libraries to requirements.txt
        with open("requirements.txt", "a") as f:
            for library in new_libraries:
                f.write(f"{library}\n")

        # Update existing libraries set
        existing_libraries.update(new_libraries)

        # Inform the user about the update
        print("requirements.txt has been updated.")

    else:
        print("---CODE IMPORT CHECK: SUCCESS---")
        # No missing libraries
        print("All required libraries are already present in requirements.txt.")

    return {
        "keys": {
            "question": question,
            "error": None,
        }
    }



def finally_execute(question):

    from langgraph.graph import END, StateGraph

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("generate", generate)  # generation solution
    workflow.add_node("check_code_imports", check_code_imports)  # check imports

    # Build graph
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "check_code_imports")
    workflow.add_conditional_edges(
        "check_code_imports",
        decide_to_finish,
        {
            "end": END,
            "generate": "generate",
        },
    )

    # Compile
    app = workflow.compile()
    config = {"recursion_limit": 20}
    app.invoke({"keys": {"question": question, "iterations": 0}}, config=config)