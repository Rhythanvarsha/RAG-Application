"""
Streamlit UI for RAC Application - Local RAG Chatbot
Medical domain focused chatbot with Chroma DB and Ollama integration
"""
import streamlit as st
import requests
import json
from typing import Dict, List
import time

# Configure Streamlit
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .chat-message {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #1f6feb;
        text-align: right;
    }
    .bot-message {
        background-color: #238636;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Embedding")
        embedding_model = st.text_input(
            "Embedding Model",
            value="llama2-embed",
            disabled=True
        )
    
    with col2:
        st.subheader("Language")
        language_model = st.text_input(
            "Language Model",
            value="llama2",
            disabled=True
        )
    
    st.divider()
    
    st.subheader("Vector Database")
    
    # Get database stats
    try:
        stats_response = requests.get(f"{API_BASE_URL}/documents/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            st.metric("ChromaDB chunks", stats.get("count", 0))
        else:
            st.metric("ChromaDB chunks", "Error")
    except:
        st.metric("ChromaDB chunks", "Offline")
    
    st.metric("Minimum relevance", 0.60)
    
    if st.button("🔄 Rebuild index", use_container_width=True):
        st.info("Index rebuild initiated...")
    
    st.divider()
    
    st.subheader("Retrieved chunks")
    retrieval_slider = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=3
    )
    
    st.divider()
    
    # Load Medical Documents
    if st.button("📚 Load Medical Documents", use_container_width=True):
        load_medical_documents()
    
    # Clear Database
    if st.button("🗑️ Clear Database", use_container_width=True):
        try:
            response = requests.delete(f"{API_BASE_URL}/documents/clear")
            if response.status_code == 200:
                st.success("Database cleared!")
                st.session_state.documents_loaded = False
        except Exception as e:
            st.error(f"Error: {e}")


def load_medical_documents():
    """Load sample medical documents into the vector database"""
    medical_docs = [
        {
            "text": "Diabetes is a chronic condition that affects blood sugar levels. Type 1 diabetes is an autoimmune disease where the pancreas cannot produce insulin. Type 2 diabetes occurs when the body becomes resistant to insulin. Symptoms include increased thirst, frequent urination, fatigue, and blurred vision.",
            "metadata": {"topic": "diabetes", "type": "medical"}
        },
        {
            "text": "Hypertension, or high blood pressure, is a serious health condition that can lead to heart disease and stroke. It's often called a silent killer because it typically has no symptoms. Risk factors include obesity, stress, salt intake, and genetics. Treatment includes lifestyle changes and medications like ACE inhibitors and beta-blockers.",
            "metadata": {"topic": "hypertension", "type": "medical"}
        },
        {
            "text": "Heart disease is the leading cause of death in many countries. It encompasses various conditions including coronary artery disease, heart failure, and arrhythmias. Risk factors include high blood pressure, high cholesterol, smoking, obesity, and diabetes. Prevention includes regular exercise, healthy diet, and stress management.",
            "metadata": {"topic": "heart_disease", "type": "medical"}
        },
        {
            "text": "Insulin is a hormone produced by the pancreas that regulates blood glucose levels. It allows cells to absorb glucose from the bloodstream for energy. The pancreas releases insulin in response to high blood sugar after meals. Dysfunction of this system leads to diabetes mellitus.",
            "metadata": {"topic": "insulin", "type": "medical"}
        },
        {
            "text": "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. Symptoms range from mild to severe and include fever, cough, fatigue, loss of taste or smell. Vaccination has been the primary prevention strategy. Treatment focuses on symptom management and supportive care.",
            "metadata": {"topic": "covid19", "type": "medical"}
        },
        {
            "text": "The immune system protects the body from infections and diseases. It includes white blood cells, antibodies, and the lymphatic system. Immunity can be innate (present from birth) or acquired (developed through exposure or vaccination). A compromised immune system increases susceptibility to infections.",
            "metadata": {"topic": "immune_system", "type": "medical"}
        },
        {
            "text": "Cancer is a disease characterized by uncontrolled cell growth and spread. There are many types including lung, breast, colon, and skin cancer. Risk factors include smoking, sun exposure, family history, and age. Treatment options include surgery, chemotherapy, radiation, and immunotherapy.",
            "metadata": {"topic": "cancer", "type": "medical"}
        },
        {
            "text": "Arthritis is inflammation of the joints causing pain, stiffness, and reduced mobility. Osteoarthritis is the most common type, resulting from wear and tear. Rheumatoid arthritis is an autoimmune disease. Treatment includes rest, physical therapy, medications, and in severe cases, surgery.",
            "metadata": {"topic": "arthritis", "type": "medical"}
        }
    ]
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/documents/add-batch",
            json={
                "documents": [doc["text"] for doc in medical_docs],
                "metadatas": [doc["metadata"] for doc in medical_docs]
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Loaded {result['count']} medical documents!")
            st.session_state.documents_loaded = True
        else:
            st.error("Failed to load documents")
    except Exception as e:
        st.error(f"Error loading documents: {e}")


# Main Chat Interface
st.title("🩺 Medical RAG Chatbot")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])
                if "retrieved_docs" in message:
                    with st.expander("📖 Retrieved context"):
                        for i, doc in enumerate(message["retrieved_docs"], 1):
                            st.write(f"**Document {i}:**")
                            st.write(doc)
                            st.divider()

# Chat input
st.divider()
col1, col2 = st.columns([0.9, 0.1])

with col1:
    user_input = st.text_input(
        "Ask about the indexed documents",
        placeholder="e.g., What is diabetes?",
        key="user_input"
    )

with col2:
    submit_button = st.button("Send", use_container_width=True)

# Process user input
if submit_button and user_input:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Show loading spinner
    with st.spinner("Processing your query..."):
        try:
            # Call RAG API
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={
                    "query": user_input,
                    "top_k": retrieval_slider
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add bot response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["generated_response"],
                    "retrieved_docs": result["retrieved_documents"]
                })
                
                # Rerun to display new messages
                st.rerun()
            else:
                st.error(f"API Error: {response.status_code}")
                st.write(response.text)
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the FastAPI server is running on port 8000")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Info footer
st.divider()
st.info(
    "💡 **Tip:** Click 'Load Medical Documents' in the sidebar to add sample medical documents, "
    "then ask questions about them!"
)
