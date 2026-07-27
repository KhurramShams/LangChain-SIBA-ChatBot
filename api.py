import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import time 
import streamlit as st
import tiktoken
from dotenv import load_dotenv

load_dotenv()
 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
# NOTE: as of LangChain 1.0, these helpers were moved out of `langchain.chains`
# into the separate `langchain-classic` package. `pip install langchain-classic`
# if you don't already have it.
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

#------------- Page Configuration ----------------------# 
st.set_page_config(page_title="SIBA RAG Assistant", page_icon=":material/smart_toy:",layout="centered")

# ------------- LLM / Chain Setup ------------------------#
 
# Check Streamlit Cloud secrets first, fallback to environment variable (.env)
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
elif "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    api_key = st.secrets["openai"]["api_key"]
else:
    api_key = os.getenv("OPENAI_API_KEY")
 
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
 
# gpt-4o-mini uses the o200k_base encoding; encoding_for_model resolves this
# correctly as long as tiktoken has the model registered.
try:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
except KeyError:
    encoding = tiktoken.get_encoding("o200k_base")
 

#------------- Load Data --------------------------------#

# Load Chroma DB from repo root
persist_directory = "./chroma_db"
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 2})


system_prompt = (
    "You are an educational chatbot for IBA Sukkur. Use the context to "
    "answer the question. If the context lacks specific details, provide "
    "what you can based on the available information and note what's "
    "missing.\n\nContext: {context}"
)
prompt_chat = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt_chat)
chain = create_retrieval_chain(retriever, question_answer_chain)

#------------- HTML+CSS -----------------------------------#
st.markdown(
    """
    <style> 
        .token-container {
            background: rgb(228, 239, 231);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
        }
         .styled-button {
            background-color: #2196F3; /* Blue */
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            text-align: center;
            display: inline-block;
            transition: 0.3s ease;
        }
        .styled-button:hover {
            background-color: #0b7dda; /* Darker Blue */
        }
    </style>
    """,
    unsafe_allow_html=True
)
#---------------Stream-Lit Fornt End Section ---------------#

if "Input_token" not in st.session_state:
    st.session_state.Input_token = 0
if "out_token" not in st.session_state:
    st.session_state.out_token = 0
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_had_sources" not in st.session_state:
    st.session_state.last_had_sources = True
    
display_chat = True
if st.session_state.Input_token > 15000:  # Example threshold
    st.warning("High token usage detected!")
    display_chat= False

if display_chat:

    # Desging App
    st.image("assets/IBA-Sukkur-logo.png", width=100)
    st.title("SIBA RAG Assistant v.0.5")
    st.write("(Beta Version)")

    #--------------- Token Section -------------------------#
    col1,col2 =st.columns(2,vertical_alignment='center')
    with col1:
         st.markdown('<div class="token-container" > <p style="font-weight: bold;">⌛ InPut Tokens</p> {} </div>'.format (st.session_state.Input_token) , unsafe_allow_html=True)
    with col2:
         st.markdown('<div class="token-container" ><p style="font-weight: bold;">⌛ Output Token</p>{}</div>'.format (st.session_state.out_token) , unsafe_allow_html=True)
    
    question = st.text_area("Ask me about IBA Sukkur! (Max 200 characters)", max_chars=200)
 
    submit_col, clear_col = st.columns(2)
    submit_clicked = submit_col.button("Submit", help="Click to get an answer")
    clear_clicked = clear_col.button("Clear Chat")
 
    if submit_clicked and question.strip():
        with st.spinner("Thinking... Please wait ⏳"):
            time.sleep(1)
            response = chain.invoke({"input": question})
            context_text = "\n".join(doc.page_content for doc in response.get("context", []))
            st.session_state.Input_token = len(encoding.encode(question + context_text))
            st.session_state.out_token = len(encoding.encode(response["answer"]))
            st.session_state.last_answer = response["answer"]
            st.session_state.last_had_sources = bool(response.get("context"))
            st.rerun()
 
    if submit_clicked and not question.strip():
        st.info("Please enter a question first.")
 
    if clear_clicked:
        st.session_state.Input_token = 0
        st.session_state.out_token = 0
        st.session_state.last_answer = None
        st.session_state.last_had_sources = True
        st.rerun()
 
    if st.session_state.last_answer is not None:
        if not st.session_state.last_had_sources:
            st.error("The documents don't contain information relevant to that question.")
        else:
            st.success(st.session_state.last_answer)
else:
    st.title("Mode Is On")