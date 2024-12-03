import streamlit as st
from itertools import zip_longest
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import openai
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Securely set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set Chroma persistent directory
os.environ["CHROMA_DB_PATH"] = "chroma_storage"

# Streamlit page configuration
st.set_page_config(page_title="Smart ChatBot For BUETK", page_icon="🤖", layout="wide")

# Center the logo and title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("buetk_logo.png", width=150)
    st.title("Smart ChatBot For BUET Khuzdar")

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.image("buetk_logo.png", width=100)

# Personalized greeting
def get_greeting():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Good Morning!"
    elif 12 <= current_hour < 18:
        return "Good Afternoon!"
    else:
        return "Good Evening!"

st.sidebar.write(f"👋 {get_greeting()}, Welcome to Smart ChatBot for BUETK!")

st.sidebar.info("Please ask your questions below.")

# Initialize session state variables for chat history
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state['past'] = []
    st.session_state['generated'] = []
    st.session_state['entered_prompt'] = ""
    st.write("Chat cleared!")

# Adjustable search depth
k_value = st.sidebar.slider("Number of relevant documents to consider:", min_value=1, max_value=10, value=3)

# Function to scrape website content
def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "li"])
        text_content = " ".join([para.get_text().strip() for para in paragraphs])
        return text_content
    except Exception as e:
        st.error(f"Error scraping the website: {e}")
        return ""

# Load data from PDFs
@st.cache_resource
def load_pdf_data():
    loader = PyPDFLoader("data/Prospectus-2021-22-1.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Scrape data from the university website
def load_website_data():
    website_url = "https://www.buetk.edu.pk"
    website_content = scrape_website(website_url)
    if website_content:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        website_documents = text_splitter.create_documents([website_content])
        return website_documents
    return []

# Combine data from PDFs and website
pdf_documents = load_pdf_data()
website_documents = load_website_data()
all_documents = pdf_documents + website_documents

# Create vector store from combined data
vector_store = Chroma.from_documents(
    all_documents,
    OpenAIEmbeddings(),
    persist_directory=os.environ["CHROMA_DB_PATH"]
)
vector_store.persist()

# Submit function for user input
def submit():
    st.session_state.entered_prompt = st.session_state.prompt_input
    st.session_state.prompt_input = ""

# Initialize ChatOpenAI model
chat = ChatOpenAI(
    temperature=0.5,
    model="gpt-3.5-turbo",
    api_key=openai.api_key,
    max_tokens=1000
)

# Function to construct messages from user and assistant history
def build_message_list():
    messages = [SystemMessage(
        content="You are AI Mentor, an expert on AI topics. Please provide accurate and polite responses."
    )]
    for user_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if user_msg is not None:
            messages.append(HumanMessage(content=user_msg))
        if ai_msg is not None:
            messages.append(AIMessage(content=ai_msg))
    return messages

# User text input for questions
st.text_input("Ask your question:", key="prompt_input", on_change=submit)

# Handle user input and generate a response
if st.session_state.entered_prompt:
    user_query = st.session_state.entered_prompt
    st.session_state.past.append(user_query)
    relevant_documents = vector_store.similarity_search(user_query, k=k_value)
    context = " ".join([doc.page_content for doc in relevant_documents])
    ai_response = chat([SystemMessage(content=f"{context}\n\n{user_query}")]).content
    st.session_state.generated.append(ai_response)

# Display conversation history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

        # Feedback mechanism
        feedback = st.radio(f"Was this response helpful?", ("👍 Yes", "👎 No"), key=f"feedback_{i}", horizontal=True)
        if feedback == "👎 No":
            st.write("Thank you for your feedback! We'll work to improve.")

# Display latest news from university
def fetch_university_news():
    news_url = "https://www.buetk.edu.pk/news"  # Example URL
    news_content = scrape_website(news_url)
    if news_content:
        st.sidebar.subheader("Latest University News")
        st.sidebar.write(news_content[:500] + "...")  # Limit display length

fetch_university_news()

# Add a footer
st.markdown("---")
st.markdown("### Developed by Student of BUET Khuzdar")
