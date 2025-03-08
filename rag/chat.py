import streamlit as st
import torch
import time
from rag import EnhancedRAG

def show_chat_page(mongo_db, user_id, rag_system=EnhancedRAG):
    """Show the main chat interface with enhanced features."""
    st.title("💬 Advanced Chat with Your Documents")
    st.markdown("Upload files and ask questions with multiple answer modes")
    
    # Sidebar for configuration and file upload
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # GPU Detection
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_info = torch.cuda.get_device_properties(0)
            st.success(f"GPU detected: {gpu_info.name} ({gpu_info.total_memory / 1024**3:.1f} GB)")
        else:
            st.warning("No GPU detected. Running in CPU mode.")
        
        # Model selection
        llm_model = st.selectbox(
            "LLM Model",
            options=["llama3.2:latest", "llama3:latest","phi3.5:3.8b","dolphin-phi:latest","samantha-mistral:latest","dolphin-mistral:latest",],
            index=0,
            key="chat_llm_model"
        )
        st.session_state.llm_model = llm_model
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ],
            index=1,
            key="chat_embedding_model"
        )
        st.session_state.embedding_model = embedding_model
        
        use_gpu = st.checkbox("Use GPU Acceleration", value=gpu_available, key="chat_use_gpu")
        st.session_state.use_gpu = use_gpu
        
        # Advanced options
        with st.expander("Advanced Options"):
            chunk_size = st.slider("Chunk Size", 100, 2000, 1000, key="chat_chunk_size")
            st.session_state.chunk_size = chunk_size
            
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, key="chat_chunk_overlap")
            st.session_state.chunk_overlap = chunk_overlap
        
        if st.button("Initialize System"):
            with st.spinner("Initializing Enhanced RAG system..."):
                st.session_state.rag = rag_system(
                    llm_model_name=llm_model,
                    embedding_model_name=embedding_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    use_gpu=use_gpu and gpu_available
                )
                st.success(f"System initialized with {embedding_model} on {st.session_state.rag.device}")
                time.sleep(1)  # Brief pause
                st.rerun()
        
        st.header("📄 Upload Documents")
        
        # Get user's notebooks for selection
        success, notebooks = mongo_db.get_notebooks(user_id)
        if success and notebooks:
            notebook_options = [("None", None)] + [(nb["name"], nb["_id"]) for nb in notebooks]
            selected_notebook = st.selectbox(
                "Add to Notebook",
                options=notebook_options,
                format_func=lambda x: x[0],
                key="upload_notebook"
            )
            selected_notebook_id = selected_notebook[1] if selected_notebook else None
            
            # Add option for custom name
            use_custom_name = st.checkbox("Use custom name", value=False)
            if use_custom_name:
                custom_name = st.text_input("Custom Document Name", placeholder="Enter custom name")
            else:
                custom_name = None
        else:
            st.write("No notebooks available. Create one in the Notebooks section.")
            selected_notebook_id = None
            custom_name = None
            
        uploaded_files = st.file_uploader("Select Files", 
                                         type=["pdf", "docx", "doc", "txt"], 
                                         accept_multiple_files=True,
                                         key="chat_file_uploader")
        
        if uploaded_files and st.button("Process Files"):
            with st.spinner("Processing files..."):
                # Save files to MongoDB if a notebook is selected
                if selected_notebook_id:
                    for file in uploaded_files:
                        file_type = "unknown"
                        if file.name.lower().endswith('.pdf'):
                            file_type = "pdf"
                        elif file.name.lower().endswith(('.docx', '.doc')):
                            file_type = "docx"
                        elif file.name.lower().endswith('.txt'):
                            file_type = "txt"
                        
                        # Save the file
                        file.seek(0)  # Reset file pointer
                        mongo_db.save_document_file(
                            file.getbuffer(),
                            file.name,
                            file_type,
                            user_id,
                            selected_notebook_id,
                            custom_name
                        )
                
                # Initialize rag system if needed
                if not st.session_state.get('rag'):
                    st.session_state.rag = rag_system(
                        llm_model_name=llm_model,
                        embedding_model_name=embedding_model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        use_gpu=use_gpu and gpu_available
                    )
                
                # Process files through RAG
                success = st.session_state.rag.process_files(
                    uploaded_files, 
                    user_id=user_id,
                    mongodb=mongo_db,
                    notebook_id=selected_notebook_id
                )
                
                if success:
                    metrics = st.session_state.rag.get_performance_metrics()
                    if metrics:
                        st.success("Files processed successfully!")
                        with st.expander("💹 Performance Metrics"):
                            st.markdown(f"**Documents processed:** {metrics['documents_processed']} chunks")
                            st.markdown(f"**Index building time:** {metrics['index_building_time']:.2f} seconds")
                            st.markdown(f"**Total processing time:** {metrics['total_processing_time']:.2f} seconds")
                            st.markdown(f"**Memory used:** {metrics['memory_used_gb']:.2f} GB")
                            st.markdown(f"**Device used:** {metrics['device']}")
                        time.sleep(1)  # Brief pause
                        st.rerun()
    
    # Mode selection in main area
    st.subheader("Select Answer Mode")
    
    if "rag_mode" not in st.session_state:
        st.session_state.rag_mode = "direct_retrieval"
    
    mode_description = {
        "direct_retrieval": "Directly retrieve answers from documents (fastest)",
        "enhanced_rag": "Enhanced RAG with multi-stage pipeline for improved answers",
        "hybrid": "Hybrid approach combining document retrieval and web search (most comprehensive)"
    }
    
    mode_cols = st.columns(3)
    with mode_cols[0]:
        direct_mode = st.button("📄 Direct Retrieval", 
                               use_container_width=True,
                               help="Fastest mode, directly uses document content to answer")
        st.caption(mode_description["direct_retrieval"])
        
    with mode_cols[1]:
        enhanced_mode = st.button("🔄 Enhanced RAG", 
                                 use_container_width=True,
                                 help="Improves answers with a multi-stage refinement process")
        st.caption(mode_description["enhanced_rag"])
        
    with mode_cols[2]:
        hybrid_mode = st.button("🌐 Hybrid Search", 
                               use_container_width=True,
                               help="Combines document content with simulated web searches")
        st.caption(mode_description["hybrid"])
    
    if direct_mode:
        st.session_state.rag_mode = "direct_retrieval"
    elif enhanced_mode:
        st.session_state.rag_mode = "enhanced_rag"
    elif hybrid_mode:
        st.session_state.rag_mode = "hybrid"
    
    st.info(f"Current mode: {st.session_state.rag_mode} - {mode_description[st.session_state.rag_mode]}")
    
    # Main chat area
    st.subheader("Ask Questions About Your Documents")
    
    # Initialize chat message history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if isinstance(message["content"], dict):
                    st.markdown(message["content"]["answer"])
                    
                    # Display mode info
                    if "mode" in message["content"]:
                        mode_name = message["content"]["mode"]
                        mode_icons = {
                            "direct_retrieval": "📄",
                            "enhanced_rag": "🔄",
                            "hybrid": "🌐"
                        }
                        icon = mode_icons.get(mode_name, "ℹ️")
                        st.caption(f"{icon} Answer mode: {mode_name}")
                    
                    if "query_time" in message["content"]:
                        st.caption(f"⏱️ Response time: {message['content']['query_time']:.2f} seconds")
                    
                    # Display pipeline info for enhanced RAG
                    if message["content"].get("mode") == "enhanced_rag" and "initial_answer" in message["content"]:
                        with st.expander("🔄 View Enhancement Process"):
                            st.subheader("Initial Answer")
                            st.markdown(message["content"]["initial_answer"])
                            st.divider()
                            st.subheader("Enhanced Answer")
                            st.markdown(message["content"]["answer"])
                    
                    # Display source info for hybrid mode
                    if message["content"].get("mode") == "hybrid":
                        if "doc_sources_count" in message["content"] and "web_sources_count" in message["content"]:
                            st.caption(f"Combined {message['content']['doc_sources_count']} document sources and {message['content']['web_sources_count']} web sources")
                    
                    # Display sources in expander
                    if "sources" in message["content"] and message["content"]["sources"]:
                        with st.expander("📄 View Sources"):
                            for i, source in enumerate(message["content"]["sources"]):
                                if source.get("file_type") == "web":
                                    st.markdown(f"**Source {i+1}: 🌐 {source['source']}**")
                                else:
                                    st.markdown(f"**Source {i+1}: 📄 {source['source']}**")
                                st.text(source["content"])
                                st.divider()
                else:
                    st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if system is initialized
        if not st.session_state.get('rag'):
            with st.chat_message("assistant"):
                message = "Please initialize the system and process documents first."
                st.markdown(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
        
        # Get response if vector store is ready
        elif hasattr(st.session_state.rag, 'doc_vector_store') and st.session_state.rag.doc_vector_store:
            with st.chat_message("assistant"):
                try:
                    with st.spinner(f"Processing with {st.session_state.rag_mode} mode..."):
                        response = st.session_state.rag.ask(
                            prompt,
                            mode=st.session_state.rag_mode,
                            user_id=user_id,
                            mongodb=mongo_db
                        )
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    if isinstance(response, dict):
                        st.markdown(response["answer"])
                        
                        # Display mode info
                        if "mode" in response:
                            mode_name = response["mode"]
                            mode_icons = {
                                "direct_retrieval": "📄",
                                "enhanced_rag": "🔄",
                                "hybrid": "🌐"
                            }
                            icon = mode_icons.get(mode_name, "ℹ️")
                            st.caption(f"{icon} Answer mode: {mode_name}")
                        
                        if "query_time" in response:
                            st.caption(f"⏱️ Response time: {response['query_time']:.2f} seconds")
                        
                        # Display pipeline info for enhanced RAG
                        if response.get("mode") == "enhanced_rag" and "initial_answer" in response:
                            with st.expander("🔄 View Enhancement Process"):
                                st.subheader("Initial Answer")
                                st.markdown(response["initial_answer"])
                                st.divider()
                                st.subheader("Enhanced Answer")
                                st.markdown(response["answer"])
                        
                        # Display source info for hybrid mode
                        if response.get("mode") == "hybrid":
                            if "doc_sources_count" in response and "web_sources_count" in response:
                                st.caption(f"Combined {response['doc_sources_count']} document sources and {response['web_sources_count']} web sources")
                        
                        # Display sources in expander
                        if "sources" in response and response["sources"]:
                            with st.expander("📄 View Sources"):
                                for i, source in enumerate(response["sources"]):
                                    if source.get("file_type") == "web":
                                        st.markdown(f"**Source {i+1}: 🌐 {source['source']}**")
                                    else:
                                        st.markdown(f"**Source {i+1}: 📄 {source['source']}**")
                                    st.text(source["content"])
                                    st.divider()
                    else:
                        st.markdown(response)
                except Exception as e:
                    error_message = f"Error generating answer: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
        else:
            with st.chat_message("assistant"):
                message = "Please upload and process documents first."
                st.markdown(message)
                st.session_state.messages.append({"role": "assistant", "content": message})