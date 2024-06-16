# pip install streamlit langchain langchain-openai beautifulsoup4 python-dotenv chromadb

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from elevenlabs import play, stream, save
from elevenlabs.client import ElevenLabs 

client = ElevenLabs(api_key="sk_e2c9a969eb8688594bd1a6dd2f926381cad891828bf76168")


load_dotenv()

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = TextLoader('Syllabi.txt')
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000000, chunk_overlap=10)
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Du bist ein KI-Studiengangsberater, welcher möglichen Studenten bei Fragen zum Studiengang Master Engineering and Management beantworten soll.")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Beantworte die Fragen freundlich und zuvorkommend und verwende den bereitgestellten Kontex, falls du die Frage nicht beantworten kannst verweise auf Lisa.Kaiser@hs-pforzheim.de:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# app config
st.set_page_config(page_title="MEM-Bot", page_icon="🤖")
st.title("MEM-Bot 📚")

# sidebar
with st.sidebar:
        #st.logo("logo.svg")
        #st.sidebar.image("logo.svg", width=100)
        st.header("Hochschule Pforzheim - Master Engineering and Management M. Sc.")

    # session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hallo ich bin dein persönlicher KI-Studiengangsberater, wie kann ich dir helfen?"),
    ]
    
    # T2S
    audio=client.generate(
        text="Hallo ich bin dein persönlicher KI-Studiengangsberater, wie kann ich dir helfen?",
        voice="ArneBanane",
        model="eleven_multilingual_v2",
        stream = True
    )
    save(audio, "audio.mp3")
    st.audio("audio.mp3", format = "audio/mp3", autoplay = True)    
    

if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_url('txt/Syllabi.txt')    

    # user input
user_query = st.chat_input("Stelle deine Fragen hier‍ 🎓")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
    
    voice_response=client.generate(
        text=response,
        voice="ArneBanane",
        model="eleven_multilingual_v2",
        stream = True
        )
    save(voice_response, "response.mp3")
    st.audio("response.mp3", format = "audio/mp3", autoplay = True)    
    

    # conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
