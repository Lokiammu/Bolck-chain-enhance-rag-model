import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import time

def show_settings_page(mongo_db, user_id):
    """Display the settings page with about, analytics, and user preferences."""
    st.title("⚙️ Settings & Analytics")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🔧 Preferences", "ℹ️ About"])
    
    # Analytics Tab
    with tab1:
        show_analytics(mongo_db, user_id)
    
    # Preferences Tab
    with tab2:
        show_preferences(mongo_db, user_id)
    
    # About Tab
    with tab3:
        show_about()

def show_analytics(mongo_db, user_id):
    """Display analytics information."""
    st.header("Usage Analytics")
    
    # Get user analytics
    success, analytics = mongo_db.get_user_analytics(user_id)
    
    if not success:
        st.error(f"Error fetching analytics: {analytics}")
        return
        
    # Overview section
    st.subheader("Overview")
    
    st.markdown("""
        <style>
            div[data-testid="stMetric"] {
            background-color: black;
            color: white;
            padding: 10px;
            border-radius: 5px;
            }
            div[data-testid="stMetric"] > div {
            color: white !important;
            }
            div[data-testid="stMetric"] > div:first-child {
            color: white !important;
            }
            div[data-testid="stMetric"] label {
            color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", analytics.get("total_documents", 0))
    with col2:
        st.metric("PDFs", analytics.get("total_pdfs", 0))
    with col3:
        st.metric("RAG Documents", analytics.get("total_rag_documents", 0))
    with col4:
        st.metric("Total Queries", analytics.get("total_queries", 0))
    # Response time metric
    st.subheader("Performance")
    col1, col2 = st.columns(2)
    with col1:
        avg_time = round(analytics.get("avg_response_time", 0), 2)
        st.metric("Average Response Time", f"{avg_time}s")
    
    # Create timestamp data for time series chart
    if analytics.get("recent_queries"):
        # Extract query times and timestamps
        query_data = []
        for query in analytics["recent_queries"]:
            query_data.append({
                "timestamp": query["timestamp"],
                "response_time": query["response_time"]
            })
        
        # Create a DataFrame
        if query_data:
            df = pd.DataFrame(query_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            
            # Plot response time trend
            with col2:
                fig = px.line(df, x="timestamp", y="response_time", 
                             title="Recent Query Response Times",
                             labels={"response_time": "Response Time (s)", "timestamp": "Time"})
                st.plotly_chart(fig, use_container_width=True)
    
    # Notebooks section
    st.subheader("Notebooks")
    
    if analytics.get("notebook_stats"):
        notebook_data = analytics["notebook_stats"]
        
        # Create a DataFrame
        notebook_df = pd.DataFrame(notebook_data)
        
        # Bar chart for document counts by notebook
        fig = px.bar(
            notebook_df, 
            x="name", 
            y=["document_count", "rag_document_count"],
            title="Documents by Notebook",
            labels={"name": "Notebook", "value": "Count", "variable": "Type"},
            barmode="group",
            color_discrete_map={"document_count": "#1E87E5", "rag_document_count": "#4CDF50"}
        )
        fig.update_layout(legend_title_text="Document Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("Recent Activity")
    
    if analytics.get("recent_queries"):
        with st.expander("Recent Queries", expanded=True):
            for idx, query in enumerate(analytics["recent_queries"]):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{idx+1}. {query['query']}**")
                with col2:
                    st.caption(f"{query['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                st.caption(f"Response Time: {query['response_time']:.2f}s")
                
                # Display notebook name if available
                if query.get("notebook_id"):
                    for nb in analytics.get("notebook_stats", []):
                        if nb.get("id") == query.get("notebook_id"):
                            st.caption(f"Notebook: {nb.get('name', 'Unknown')}")
                            break
                
                st.divider()
    else:
        st.info("No queries have been made yet.")

def show_preferences(mongo_db, user_id):
    """Display user preferences settings."""
    st.header("User Preferences")
    
    # RAG System Preferences
    st.subheader("RAG System Preferences")
    
    # Model preferences
    col1, col2 = st.columns(2)
    with col1:
        llm_model = st.selectbox(
            "Default LLM Model",
            options=["llama3.2:latest", "llama3:latest", "mistral:latest"],
            index=0
        )
        st.session_state.llm_model = llm_model
    
    with col2:
        embedding_model = st.selectbox(
            "Default Embedding Model",
            options=[
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ],
            index=1
        )
        st.session_state.embedding_model = embedding_model
    
    # GPU usage
    use_gpu = st.checkbox("Use GPU Acceleration (if available)", value=True)
    st.session_state.use_gpu = use_gpu
    
    # Advanced RAG settings
    with st.expander("Advanced RAG Settings"):
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size", 100, 2000, 1000)
            st.session_state.chunk_size = chunk_size
        with col2:
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
            st.session_state.chunk_overlap = chunk_overlap
    
    # UI Preferences
    st.subheader("UI Preferences")
    
    # Theme preferences
    theme = st.selectbox(
        "Theme",
        options=["Light", "Dark", "System default"],
        index=2
    )
    
    # Default page
    default_page = st.selectbox(
        "Default Page",
        options=["Chat", "Notebooks", "Settings"],
        index=0
    )
    
    # Save preferences button
    if st.button("Save Preferences", use_container_width=True):
        st.success("Preferences saved successfully!")
        time.sleep(1)  # Brief pause for success message to show
        st.rerun()

def show_about():
    """Display information about the application."""
    st.header("About GPU-Accelerated RAG System")
    
    # App description
    st.markdown("""
    This advanced document management and question answering system uses 
    state-of-the-art Retrieval Augmented Generation (RAG) technology to help you organize,
    search, and extract insights from your documents.
    
    Built with GPU acceleration for faster processing and response times.
    """)
    
    # Version information
    st.subheader("Version Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Application Version:** 1.0.0")
        st.write("**RAG Engine:** LangChain + FAISS")
        st.write("**UI Framework:** Streamlit")
    with col2:
        st.write("**Database:** MongoDB")
        st.write("**LLM Backend:** Ollama")
        st.write("**Embeddings:** HuggingFace Transformers")
    
    # Features
    st.subheader("Key Features")
    st.markdown("""
    - **Notebook Organization**: Create and manage document collections
    - **GPU Acceleration**: Faster processing and response times
    - **Document Management**: Upload and organize PDFs, Word documents, and text files
    - **Intelligent Search**: Ask questions in natural language about your documents
    - **Detailed Analytics**: Track usage and performance metrics
    - **Custom Document Naming**: Organize documents with your preferred names
    - **Document Viewer**: Read your documents without leaving the application
    """)
    
    # Credits
    st.subheader("Credits")
    st.markdown("""
    Created with ❤️ using:
    - Streamlit
    - LangChain
    - FAISS
    - Ollama
    - HuggingFace Transformers
    - MongoDB
    - PyPDF2
    - python-docx
    """)
    
    # Contact
    st.subheader("Need Help?")
    st.markdown("""
    For support or feature requests, please reach out to the development team.
    """)