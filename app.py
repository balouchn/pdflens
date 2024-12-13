import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import openai

# Load environment variables from .env file
load_dotenv()

# Fetch OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded properly
if not openai_api_key:
    st.error("OPENAI_API_KEY is missing in the .env file.")
    st.stop()  # Stop execution if API key is missing
else:
    openai.api_key = openai_api_key  # Set the API key globally for OpenAI API calls

# Streamlit app starts here
def main():
    # Set page configuration
    st.set_page_config(page_title="PDFLens", page_icon=":page_with_curl:")

    # Inject custom CSS for Crimson Red Theme
    st.markdown("""
        <style>
            body {
                background-color: #DC143C;
                color: white;
                font-family: 'Poppins', sans-serif;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #ffffff;
            }
            .stButton>button {
                background-color: #8B0000;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
                cursor: pointer;
            }
            .stButton>button:hover {
                background-color: #A52A2A;
            }
            .stTextInput input {
                background-color: #8B0000;
                color: white;
                border-radius: 5px;
                padding: 10px;
                border: 2px solid #A52A2A;
            }
            .stTextInput input:focus {
                border-color: #DC143C;
            }
            .stMarkdown {
                color: white;
            }
            .chat-message {
                margin-bottom: 10px;
                padding: 10px;
                border-radius: 8px;
                background-color: #8B0000;
                color: white;
                border: 1px solid #A52A2A;
            }
            .avatar img {
                width: 50px;
                height: 50px;
                border-radius: 50%;
                border: 2px solid #DC143C;
            }
            .message {
                margin-left: 10px;
                display: inline-block;
            }
            .chat-message.bot {
                background-color: #800000;
                border-color: #A52A2A;
            }
            .chat-message.user {
                background-color: #A52A2A;
                border-color: #800000;
            }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Initialize chat history
    if "conversation" not in st.session_state:
        st.session_state.conversation = None  # Initialize conversation chain

    # PDF Text Extraction Function
    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    # Text Chunking Function
    def get_text_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    # Vector Store Setup
    def get_vectorstore(text_chunks):
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Ensure API key is passed here
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            return vectorstore
        except Exception as e:
            st.error(f"Error while setting up vectorstore: {e}")
            return None

    # Conversation Chain Setup
    def get_conversation_chain(vectorstore):
        try:
            llm = ChatOpenAI(openai_api_key=openai_api_key)  # Ensure API key is passed here
            memory = ConversationBufferMemory(
                memory_key='chat_history', return_messages=True)
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory
            )
            return conversation_chain
        except Exception as e:
            st.error(f"Error while setting up conversation chain: {e}")
            return None

    # Handle User Input and Conversation
    def handle_userinput(user_question, conversation):
        response = conversation.invoke({'question': user_question})
        return response['answer']

    # Set title for the page
    st.title("PDFLens")

    # Sidebar for uploading PDFs
    pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
    if pdf_docs:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.vectorstore = vectorstore  # Save the vectorstore in session state

    # Add a "Process" button
    process_button = st.button("Process PDF & Start Chat")

    if process_button:
        # Only process when the button is clicked
        if pdf_docs and st.session_state.vectorstore:
            # Get the conversation chain once PDFs are processed
            conversation_chain = get_conversation_chain(st.session_state.vectorstore)
            if conversation_chain:
                st.session_state.conversation = conversation_chain
                st.session_state.chat_history = []  # Reset chat history when processing starts
                st.success("PDF processed, conversation ready!")
            else:
                st.error("Failed to set up conversation chain.")
        else:
            st.error("Please upload PDFs first.")

    # User Question
    user_question = st.text_input("Ask a question:")
    if user_question and st.session_state.conversation:
        response = handle_userinput(user_question, st.session_state.conversation)
        st.session_state.chat_history.append({"user": user_question, "bot": response})

    # Display Chat History
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.markdown(f"<div class='chat-message user'>{chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-message bot'>{chat['bot']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

