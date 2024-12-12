from typing import List, Tuple, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.ai import AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

import os
import uuid
from dotenv import load_dotenv

import streamlit as st
from streamlit_ace import st_ace, LANGUAGES

from utils.graph_builders import build_graph, build_pyrepl_graph
from utils.prompt_templates import *

load_dotenv(dotenv_path='prod.env')

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_PROJECT"] = "Code Analysis AI Tool"

thread_id = str(uuid.uuid4())

st.set_page_config(page_title="Code Generation and Analysis AI Tool", layout="wide")

st.sidebar.title("Navigation & Settings")
page = st.sidebar.selectbox("Choose a task", ["Coding Playground", "Codebase Analysis"])

llm = ChatOpenAI(model_name = 'gpt-4o')

if "editor_code" not in st.session_state:
    st.session_state.editor_code = ""
if "last_content" not in st.session_state:
    st.session_state.last_content = ""
if "suggestion" not in st.session_state:
    st.session_state.suggestion = ""
if "ace_key" not in st.session_state:
    st.session_state.ace_key = 0
        
if page == "Coding Playground":
    def stream_ca_graph_updates(user_input: str):
        graph = build_pyrepl_graph(llm)
        events = graph.stream(
            {"messages": ("user", user_input)}, stream_mode="values"
        )
        
        last_ai_message = None
        for event in events:
            for message in event.get("messages", []):
                if isinstance(message, AIMessage):
                    last_ai_message = message.content
        
        st.markdown(f"{last_ai_message}")
        
    selected_language = st.sidebar.selectbox("Language Mode for Editor", options=LANGUAGES, index=121)

    st.subheader("Playground for Coding")
    st.write("Write your code in the editor below. Click Apply or **Ctrl+Enter** to see the AI's suggestion.")

    chain = auto_complete_prompt | llm
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Code Editor")
        content = st_ace(
            value=st.session_state.editor_code,
            placeholder=f"Let's write some {selected_language} code together!",
            language=selected_language,
            theme="ambiance",
            keybinding="vscode",
            font_size=16,
            tab_size=4,
            auto_update=False,
            min_lines=24,
            key=f"ace-{st.session_state.ace_key}",
        )
    
    with col2:
        st.subheader("Suggestions")
        
        if selected_language == "python":
            c1, c2 = st.columns([2, 1.2])
            with c1:
                insert_clicked = st.button("Insert Suggestion")
            with c2:
                execcode_clicked = st.button("Execute Code")
        else:
            insert_clicked = st.button("Insert Suggestion")
        
        if not insert_clicked and not execcode_clicked and content.strip():
            if content != st.session_state.last_content:
                with st.spinner("Generating suggestion..."):
                    suggestion = chain.invoke({"code": content, "language": selected_language}).content.strip()
                    suggestion = suggestion.replace(f"```{selected_language}", "").replace("```", "")
                    st.session_state.suggestion = suggestion
                    st.session_state.last_content = content
            else:
                st.info("No changes detected in the editor. Modify the code and click on Apply to generate a new suggestion.")
        elif not content.strip():
            st.warning("Please write some code in the editor first!")
        
        if insert_clicked:
            if st.session_state.suggestion:
                st.session_state.editor_code = content + "\n" + st.session_state.suggestion
                st.session_state.last_content = st.session_state.editor_code
                st.session_state.suggestion = ""
                st.session_state.ace_key += 1
                st.rerun()
            else:
                st.warning("No suggestion available to insert!")
                
        if execcode_clicked:
            if content.strip():
                st.session_state.suggestion = ""
                try:
                    with st.spinner("Executing code..."):
                        stream_ca_graph_updates(content)
                except:
                    user_input = "What do you know about LangGraph?"
                    print("User: " + user_input)
                    stream_ca_graph_updates(user_input)
            else:
                st.warning("Please write some code in the editor first!")

        if st.session_state.suggestion:
            st.write("Auto-completion Suggestion:")
            st.markdown(f"In {selected_language}\n{st.session_state.suggestion}")
            
elif page == "Codebase Analysis":
    st.subheader("Analyze Uploaded Codebase")
    st.write("Start all kinds of analysis on your codebase by entering the folder path below:")
    
    def get_all_files(folder_path):
        all_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        return all_files

    @st.cache_resource
    def load_and_split_code(folder_path):
        global thread_id
        code_extensions: Union[List[str], Tuple[str], str] = [
            "**/*.py",
            "**/*.js",
            "**/*.ts",
            "**/*.jsx",
            "**/*.tsx",
            "**/*.java",
            "**/*.cpp",
            "**/*.c",
            "**/*.cs",
            "**/*.rb",
            "**/*.go",
            "**/*.php",
            "**/*.html",
            "**/*.css",
            "**/*.swift",
            "**/*.kt",
            "**/*.rs",
            "**/*.dart",
            "**/*.sh",
            "**/*.bat",
            "**/*.sql",
            "**/*.pl",
            "**/*.pm",
            "**/*.rb",
            "**/*.lua",
            "**/*.r",
            "**/*.jl",
            "**/*.hs",
        ]
        loader = DirectoryLoader(folder_path, glob=code_extensions)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100, length_function=len)
        thread_id = str(uuid.uuid4())
        return text_splitter.split_documents(documents)

    @st.cache_resource
    def create_embeddings(_texts):
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(_texts, embeddings)
        return vector_store

    selected_file = None
    enable_analysis = False
    file_paths = []
    
    code_folder = st.text_input("Enter the path to the folder containing your code files:")
    if code_folder:
        if os.path.exists(code_folder) and os.path.isdir(code_folder):
            st.success(f"Folder `{code_folder}` selected.")
            file_paths = get_all_files(code_folder)
            file_names = []
            file_names.append(code_folder)
            file_names.extend([os.path.relpath(path, code_folder) for path in file_paths])
            st.success(f"Loaded {len(file_paths)} files from `{code_folder}`.")
            
            enable_analysis = True
        else:
            st.error("Invalid folder path. Please check and try again.")

    if enable_analysis:
        with st.spinner("Loading and splitting code files..."):
            print(code_folder)
            texts = load_and_split_code(code_folder)

        with st.spinner("Creating embeddings..."):
            vector_store = create_embeddings(texts)

    def stream_graph_updates(user_input: str):
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        graph = build_graph(llm, vector_store)
        events = graph.stream(
            {"messages": ("user", user_input)}, config, stream_mode="values"
        )
        if events is None:
            st.write("No events found.")
            return
        
        last_ai_message = None
        for event in events:
            for message in event.get("messages", []):
                if isinstance(message, AIMessage):
                    last_ai_message = message.content
        
        if last_ai_message:
            st.markdown(f"{last_ai_message}")
        else:
            st.warning("No AIMessage content found.")
        
    if enable_analysis:
        st.write("")
        st.write("")
        st.subheader("Query Your Codebase")
        st.write("To analyze your local codebase, prefix your query with '@' and select a file or directory. For example, '@Optimize the upload page of my web app'.")
        query = st.text_input("Enter your question or search term:")
        command_input = query
        
        if query:
            if query.startswith("@") and file_names:
                selected_file = st.selectbox("Select a file/directory", file_names)
            elif len(query) > 0:
                try:
                    with st.spinner("Processing query..."):
                        stream_graph_updates(query)
                except:
                    user_input = "What do you know about LangGraph?"
                    print("User: " + user_input)
                    stream_graph_updates(user_input)
            else:
                st.warning("Please enter a valid query.")
        
        if st.button("Execute Command"):
            if not query.startswith("@") and len(query) > 0:
                try:
                    with st.spinner("Processing query..."):
                        stream_graph_updates(query)
                except:
                    user_input = "What do you know about LangGraph?"
                    print("User: " + user_input)
                    stream_graph_updates(user_input)
            elif len(query) == 0 or (query.startswith("@") and query.strip() == "@"):
                st.warning("Please enter a valid query.")
            else:
                file_path = os.path.join(code_folder, selected_file)
                
                if selected_file and os.path.isfile(file_path) and command_input:
                    mod_cmd = command_input[1:].strip()
                    with open(file_path, "r") as f:
                        file_content = f.read()
                    
                    chain = per_file_analysis_prompt | llm
                    
                    with st.spinner("Processing command..."):
                        st.markdown(f"{chain.invoke({"command_input": mod_cmd, "file_content": file_content}).content}")
                    
                elif selected_file and os.path.isdir(selected_file) and command_input:
                    try:
                        with st.spinner("Processing on relevant code..."):
                            stream_graph_updates(command_input)
                    except:
                        user_input = "What do you know about LangGraph?"
                        print("User: " + user_input)
                        stream_graph_updates(user_input)
                else:
                    st.warning("Please enter a valid command and select a file/directory.")
