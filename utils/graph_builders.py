from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from langchain_core.runnables import Runnable
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State):
        for i in range(10):
            result = self.runnable.invoke(state)
            
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break

        return {"messages": result}

def handle_tool_error(state) -> dict:
    """
    Function to handle errors that occur during tool execution.
    
    Args:
        state (dict): The current state of the AI agent, which includes messages and tool call details.
    
    Returns:
        dict: A dictionary containing error messages for each tool that encountered an issue.
    """
    error = state.get("error")
    
    tool_calls = state["messages"][-1].tool_calls
    
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    """
    Function to create a tool node with fallback error handling.
    
    Args:
        tools (list): A list of tools to be included in the node.
    
    Returns:
        dict: A tool node that uses fallback behavior in case of errors.
    """
    
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)],
        exception_key="error"
    )
    
def build_pyrepl_graph(llm):
    repl = PythonREPL()
    
    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                ''' You are a coding assistant specialized in executing Python code. Your sole task is to determine whether the input is valid and complete Python code, execute it using the `python_repl_tool`, and provide the appropriate response.

                Input Handling:
                1. Valid and Complete Python Code:
                - Execute the input using the `python_repl_tool`.
                - Return the execution result in the format:
                    ```
                    Successfully executed:
                    <Code>
                    Output:
                    <Execution Result>
                    ```
                - If execution fails, return an error message in the format:
                    ```
                    Failed to execute:
                    <Code>
                    Error:
                    <Error Message>
                    ```

                2. Invalid Python Code:
                - If the input is Python code but contains syntax or other errors that prevent execution, return an appropriate error message as provided by the `python_repl_tool`.

                3. Code in Any Other Language:
                - If the input is not Python code, warn the user that only Python code execution is supported, in the format:
                    ```
                    Unsupported language:
                    The input appears to be code in a language other than Python. This assistant only supports Python code execution.
                    ```

                Example Scenarios:

                - Valid Python Code:
                Input: `"print(5 + 10)"`
                Response:
                Successfully executed:
                print(5 + 10)
                Output:
                15
                ```

                - Invalid Python Code:
                Input: `"print(5 + )"`
                Response:
                ```
                Failed to execute:
                print(5 + )
                Error:
                SyntaxError: invalid syntax
                ```

                - Unsupported Language:
                Input: `"function greet(name) {{ return `Hello, ${{name}}`; }}"`
                Response:
                ```
                Unsupported language:
                The input appears to be code in a language other than Python. This assistant only supports Python code execution.
                ```
                '''
            ),
            ("placeholder", "{messages}"),
        ]
    )

    @tool
    def python_repl_tool(
        code: Annotated[str, "The Python code to execute. Ensure that the code is syntactically correct and any required libraries are imported."],
    ):
        """
        Execute Python code dynamically using a Python REPL environment.

        Purpose:
        - This tool allows users to run Python code snippets in real-time.
        - Useful for generating outputs, performing computations, or testing small blocks of Python code.

        Usage:
        - Input: A string containing the Python code to execute.
        - Example:
            ```
            code = "print(5 + 3)"
            result = python_repl_tool(code)
            ```
        - To view the output of a value, explicitly use `print(...)` in the code. For example:
            ```
            print("Hello, World!")
            print(10 + 20)
            ```

        Output:
        - On successful execution:
            ```
            Successfully executed:
            ```python
            <Code>
            ```
            Stdout: <Execution Result>
            ```
        - On failure:
            - Returns an error message with details about the exception encountered.

        Notes:
        - Ensure the code is safe to execute. The tool does not provide sandboxing for untrusted input.
        """
        try:
            result = repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return result

    assistant_tools = [
        python_repl_tool
    ]

    assistant_runnable = primary_assistant_prompt | llm.bind_tools(assistant_tools)
    builder = StateGraph(State)
    builder.add_node("assistant", Assistant(assistant_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(assistant_tools))
    
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    
    graph = builder.compile()
    
    return graph

def build_graph(llm, vector_store):    
    @tool
    def code_analysis_tool(query: str) -> float:
        """
        Tool to analyze the user's codebase and return relevant results or matches based on the input query, 
        along with references to the parts of the codebase used to generate the response.

        Args:
            query (str): The user's query or command related to their codebase. This can include requests like 
                        "Find all functions that use 'database_connection'", "List all classes in module X", 
                        or "Generate documentation for module Y".
        
        Returns:
            str: A formatted string containing:
                - The analysis result or answer to the query.
                - References to the relevant sections of the codebase used in the analysis.
        """
        
        code_analysis_prompt_template = """
        You are a highly skilled and knowledgeable coding assistant tasked with analyzing codebases and providing precise, actionable responses to user queries. Your goal is to assist the user by leveraging the provided context and question to generate accurate, insightful, and practical answers.

        Instructions:
        1. Use the provided context to analyze and address the user's query thoroughly. 
        2. If the context lacks sufficient information to answer the query, state that you don't know and avoid making up information.
        3. When answering:
        - Ensure your response is structured, clear, and concise.
        - Directly address the query with actionable suggestions and explanations where applicable.
        - Include code snippets formatted in the appropriate programming language if relevant. Each snippet must have a purpose clearly indicated above it.

        Formatting Rules:
        - Begin your response with a direct answer or actionable suggestion.
        - Include well-formatted and syntax-highlighted code snippets if applicable.
        - Include references to the provided context where applicable to support your answer.
        
        Question: {question}
        Context:
        ---------
        {context}
        ---------
        """

        try:
            relevant_chunks = vector_store.similarity_search_with_score(query, k=3)
            chunk_docs = [chunk[0] for chunk in relevant_chunks]

            document_prompt = PromptTemplate(
                input_variables=["source", "page_content"],
                template="Source: {source}\nPage Content: {page_content}"
            )
            prompt = PromptTemplate(
                template=code_analysis_prompt_template,
                input_variables=["context", "question"]
            )
            
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            
            chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_prompt=document_prompt,
                document_variable_name="context",
            )
            results = chain({"input_documents": chunk_docs, "question": query})
            text_reference = "\n".join([doc.metadata['source'] for doc in chunk_docs])
            result = f"{results["output_text"]}\n\nReferences: {text_reference}"
        except BaseException as e:
            result = f"Failed to execute. Error: {repr(e)}"
        return result
    
    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''You are a coding assistant capable of answering general programming questions and analyzing local codebases.
                
                To analyze local codebases, users must prefix their queries with "@". Handle queries as follows:

                1. If the query starts with "@", it indicates a request to analyze the local codebase. Call the `code_analysis_tool` with the query (excluding the "@") to process the request. Ensure that the response includes:
                - Relevant insights or suggestions based on the analysis.
                - References to specific files, functions, or code snippets from the codebase wherever applicable.

                2. If the query does not start with "@", handle it as a general programming question and respond directly using your knowledge. Whenever possible:
                - Include references to relevant documentation, libraries, or official resources.
                - Provide examples or links to reliable external sources if the information benefits the user.

                Example Queries:
                - "@Optimize the upload page of my web app" -> Call the `code_analysis_tool`.
                - "Give me the Python code for merge sort" -> Answer directly as a general programming question.

                If you are unsure about the type of query or the user's intent, ask for clarification. Always provide a helpful and concise response.

                Rules:
                - Queries starting with "@": Route to the `code_analysis_tool` and include references from the codebase.
                - Other queries: Provide a direct response as the assistant, including external references where applicable.

                Output Guidelines:
                - Always include references if they enhance the quality of the answer.
                - Ensure outputs are well-structured, concise, and easy to understand.
                - If no relevant action can be taken, politely inform the user.

                Example Output with References:
                ```
                Helpful Answer:
                To optimize the upload page, consider implementing the following changes:
                1. Improve form validation using hooks (Refer to `app/upload/form.js`, line 42).
                2. Optimize API calls to reduce latency (Refer to `services/api/upload.js`, function `uploadFile`).

                References:
                - app/upload/form.js, line 42
                - services/api/upload.js, function `uploadFile`
                ```
                ''',
            ),
            ("placeholder", "{messages}"),
        ]
    )

    assistant_tools = [
        code_analysis_tool
    ]

    assistant_runnable = primary_assistant_prompt | llm.bind_tools(assistant_tools)
    builder = StateGraph(State)
    builder.add_node("assistant", Assistant(assistant_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(assistant_tools))
    
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    return graph
