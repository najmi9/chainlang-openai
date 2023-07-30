import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.document_loaders import UnstructuredURLLoader
from typing import List
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_website_chunks(urls: List[str]) -> List[str]:
    loader = UnstructuredURLLoader(urls)
    chunks = loader.load_and_split()

    return chunks

def handle_userinput(user_question):
    if "agent" in st.session_state:
        response = st.session_state.agent.run(user_question)
        st.write(user_template.replace(
                "{{MSG}}", user_question), unsafe_allow_html=True)
        st.write(bot_template.replace(
                "{{MSG}}", response), unsafe_allow_html=True)

        return

    response = st.session_state.conversation({'question': user_question})

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat your data",
                       page_icon=":file:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "agent" not in st.session_state:
        st.session_state.agent = None

    st.header("Chat with your data :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your PDF documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process my PDFs", type='primary') and pdf_docs is not None:
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

        st.subheader("Websites to talk with")
        website_url = st.text_input(
            "Enter the URLs separated by a comma:",
            placeholder="https://google.com, https://..."
        )
        if st.button('Load my website', type='primary') and website_url is not None:
            with st.spinner('Processing'):
                urls = website_url.split(', ')
                text_chunks = get_website_chunks(urls)
                text_chunks = list(map(lambda document: document.page_content, text_chunks))
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore
                )

        st.subheader("Your CSV files")
        csv_files = st.file_uploader("Upload a CSV file", type="csv", accept_multiple_files=True)
        if st.button('Process my CSV files', type='primary'):
            with st.spinner('Processing'):
                llm = OpenAI(temperature=0)
                paths = list(map(lambda file: file.name, csv_files))
                agent = create_csv_agent(
                    llm, paths, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                )
                st.session_state.agent = agent

if __name__ == '__main__':
    main()
