# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from vector_store_utils import create_or_load_vector_store

# Load environment variables from .env file
load_dotenv()

# Function to load the RAG chain
# We use @st.cache_resource to load this only once
@st.cache_resource
def load_chain():
    # Ensure the OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Please set it in your .env file or secrets.")
        st.stop()
    
    # Create or load the vector store
    with st.spinner("Loading vector store and setting up RAG chain... This may take a few minutes on first run."):
        vector_store = create_or_load_vector_store()
        retriever = vector_store.as_retriever()
        
        # Set up the language model and the RAG chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
    return qa_chain

# --- Streamlit App UI ---

st.set_page_config(page_title="Chat with Walmart Sales Data", page_icon="📈")
st.title("📈 Chat with Walmart Sales Data")

# --- UPDATED AND MORE DESCRIPTIVE TEXT ---
st.write(
    "Ask any question about the sales data for **45 Walmart stores**! "
    "You can ask about:"
)
st.markdown("""
- **Weekly Sales**
- **Temperature**
- **Fuel Price**
- **CPI (Consumer Price Index)**
- **Unemployment Rate**
""")
st.info("Example questions:\n"
        "- *'What were the weekly sales for Store 20 on 2012-02-10?'*\n"
        "- *'Which store had the highest sales during a holiday week?'*\n"
        "- *'What was the unemployment rate when the fuel price was at its lowest?'*")


# Load the RAG chain
try:
    qa_chain = load_chain()
except Exception as e:
    st.error(f"Failed to load the application. Error: {e}")
    st.stop()


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with the sales data today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke(prompt)
                st.markdown(response['result'])
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response['result']})
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})