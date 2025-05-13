import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Force use of pysqlite3 (for ChromaDB compatibility)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import time 
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
import tiktoken
from langchain_community.callbacks.manager import get_openai_callback  # Updated import
from langchain_chroma import Chroma  # Updated Chroma import
from langchain_openai import OpenAIEmbeddings
#------------- Page Configuration ----------------------# 
st.set_page_config(page_title="SIBA CHAT-BOT", page_icon=":material/smart_toy:",layout="centered")

#------------- LLM Function -----------------------------#

api_key = st.secrets["openai"]["api_key"]

llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
encoding = tiktoken.encoding_for_model("text-davinci-003")

#------------- Load Data --------------------------------#

# Load Chroma DB from repo root
persist_directory = "./chroma_db"
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 2})

template ="""
You are an educational chatbot for IBA Sukkur. Use the context to answer the question. If the context lacks specific details, provide what you can based on the available information and note what’s missing.
Context: {context}
Question: {question}
Answer:
"""
prompt_chat=PromptTemplate(template=template,input_variables=["context","question"])

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_chat},
    return_source_documents=True
)

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


Disply= True
if "Input_token" not in st.session_state:
    st.session_state.Input_token = 0
if "out_token" not in st.session_state:
    st.session_state.out_token = 0

if st.session_state.Input_token > 15000:  # Example threshold
    st.warning("High token usage detected!")
    Disply= Flase

if Disply:
    token=10
    # Desging App
    st.image("assets/IBA-Sukkur-logo.png", width=100)
    st.title("SIBA Chat-Bot v.0.5")
    st.write("(Beta Version)")
    #--------------- Token Section -------------------------#
    col1,col2 =st.columns(2,vertical_alignment='center')
    with col1:
         st.markdown('<div class="token-container" > <p style="font-weight: bold;">⌛ InPut Tokens</p> {} </div>'.format (st.session_state.Input_token) , unsafe_allow_html=True)
    with col2:
         st.markdown('<div class="token-container" ><p style="font-weight: bold;">⌛ Output Token</p>{}</div>'.format (st.session_state.out_token) , unsafe_allow_html=True)
    
    question =st.text_area("Ask me about IBA Sukkur! (Max 200 characters)")

    if st.button("Submit",help="Click to get an answer"):
        with st.spinner("Thinking... Please wait ⏳"):
            time.sleep(3)  # Simulate processing time
            with get_openai_callback() as cb:
                response = chain.invoke({"query": question})
                st.session_state.Input_token = cb.prompt_tokens
                st.session_state.out_token = cb.completion_tokens 


        if not response["source_documents"]:
            st.error("The document doesn't provide information about the weather at IBA Sukkur.")
        else:
            st.success(response["result"])
            
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
else:
    st.title("Mode Is On")

