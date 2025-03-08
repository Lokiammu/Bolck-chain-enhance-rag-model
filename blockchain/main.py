# main.py
import os
import tempfile
import shutil
import PyPDF2
import streamlit as st
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import time
import psutil
import uuid
import atexit
from blockchain_utils import BlockchainManager


class BlockchainEnabledRAG:
    def __init__(self, 
                 llm_model_name="llama3.2:latest",
                 embedding_model_name="sentence-transformers/all-mpnet-base-v2",
                 chunk_size=1000,
                 chunk_overlap=200,
                 use_gpu=True,
                 use_blockchain=False,
                 blockchain_url="http://localhost:8545",
                 contract_address=None,
                 private_key=None):
        """
        Initialize the GPU-efficient RAG system with blockchain integration.
        
        Args:
            llm_model_name: The Ollama model for text generation
            embedding_model_name: The HuggingFace model for embeddings
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            use_gpu: Whether to use GPU acceleration
            use_blockchain: Whether to enable blockchain verification
            blockchain_url: URL of the blockchain node
            contract_address: Address of the deployed RAG Document Verifier contract
            private_key: Private key for signing transactions
        """
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.use_blockchain = use_blockchain
        
        # Device selection for embeddings
        self.device = "cuda" if self.use_gpu else "cpu"
        st.sidebar.info(f"Using device: {self.device}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": self.device}
        )
        
        # Initialize LLM
        self.llm = OllamaLLM(model=llm_model_name)
        
        # Initialize vector store
        self.vector_store = None
        self.documents_processed = 0
        
        # Monitoring stats
        self.processing_times = {}
        
        # Initialize blockchain manager if enabled
        self.blockchain = None
        if use_blockchain:
            try:
                self.blockchain = BlockchainManager(
                    blockchain_url=blockchain_url,
                    contract_address=contract_address,
                    private_key=private_key
                )
                st.sidebar.success("Blockchain connection established")
            except Exception as e:
                st.sidebar.error(f"Failed to connect to blockchain: {str(e)}")
                self.use_blockchain = False

    def process_pdfs(self, pdf_files):
        """Process PDF files, create a vector store, and verify documents on blockchain."""
        all_docs = []
        
        with st.status("Processing PDF files...") as status:
            # Create temporary directory for file storage
            temp_dir = tempfile.mkdtemp()
            st.session_state['temp_dir'] = temp_dir
            
            # Monitor processing time and memory usage
            start_time = time.time()
            
            # Track memory before processing
            mem_before = psutil.virtual_memory().used / (1024 * 1024 * 1024)  # GB
            
            # Process each PDF file
            for i, pdf_file in enumerate(pdf_files):
                try:
                    file_start_time = time.time()
                    
                    # Save uploaded file to temp directory
                    pdf_path = os.path.join(temp_dir, pdf_file.name)
                    with open(pdf_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    
                    status.update(label=f"Processing {pdf_file.name} ({i+1}/{len(pdf_files)})...")
                    
                    # Extract text from PDF
                    text = ""
                    with open(pdf_path, "rb") as f:
                        pdf = PyPDF2.PdfReader(f)
                        for page_num in range(len(pdf.pages)):
                            page = pdf.pages[page_num]
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n\n"
                    
                    # Create documents
                    docs = [Document(page_content=text, metadata={"source": pdf_file.name})]
                    
                    # Split documents into chunks
                    split_docs = self.text_splitter.split_documents(docs)
                    
                    all_docs.extend(split_docs)
                    
                    # Verify document on blockchain if enabled
                    if self.use_blockchain and self.blockchain:
                        try:
                            # Create a unique document ID
                            document_id = f"{pdf_file.name}_{uuid.uuid4().hex[:8]}"
                            
                            # Verify document on blockchain
                            status.update(label=f"Verifying {pdf_file.name} on blockchain...")
                            verification = self.blockchain.verify_document(document_id, pdf_path)
                            
                            if verification['status'] == 1:  # Success
                                st.sidebar.success(f"✅ {pdf_file.name} verified on blockchain")
                                st.sidebar.info(f"Transaction: {verification['tx_hash'][:10]}...")
                                
                                # Add blockchain metadata to documents
                                for doc in split_docs:
                                    doc.metadata["blockchain"] = {
                                        "verified": True,
                                        "document_id": document_id,
                                        "document_hash": verification["document_hash"],
                                        "tx_hash": verification["tx_hash"],
                                        "block_number": verification["block_number"]
                                    }
                            else:
                                st.sidebar.warning(f"❌ Failed to verify {pdf_file.name} on blockchain")
                        except Exception as e:
                            st.sidebar.error(f"Blockchain verification error: {str(e)}")
                    
                    file_end_time = time.time()
                    processing_time = file_end_time - file_start_time
                    
                    st.sidebar.success(f"Processed {pdf_file.name}: {len(split_docs)} chunks in {processing_time:.2f}s")
                    self.processing_times[pdf_file.name] = {
                        "chunks": len(split_docs),
                        "time": processing_time
                    }
                    
                except Exception as e:
                    st.sidebar.error(f"Error processing {pdf_file.name}: {str(e)}")
            
            # Create vector store if we have documents
            if all_docs:
                status.update(label="Building vector index using GPU...")
                try:
                    # Record the time taken to build the index
                    index_start_time = time.time()
                    
                    # Create the vector store using FAISS
                    self.vector_store = FAISS.from_documents(all_docs, self.embeddings)
                    
                    index_end_time = time.time()
                    index_time = index_end_time - index_start_time
                    
                    # Track memory after processing
                    mem_after = psutil.virtual_memory().used / (1024 * 1024 * 1024)  # GB
                    mem_used = mem_after - mem_before
                    
                    total_time = time.time() - start_time
                    
                    status.update(label=f"Completed processing {len(all_docs)} chunks in {total_time:.2f}s", state="complete")
                    
                    # Save performance metrics
                    self.processing_times["index_building"] = index_time
                    self.processing_times["total_time"] = total_time
                    self.processing_times["memory_used_gb"] = mem_used
                    self.documents_processed = len(all_docs)
                    
                    return True
                except Exception as e:
                    st.error(f"Error creating vector store: {str(e)}")
                    status.update(label="Error creating vector store", state="error")
                    return False
            else:
                status.update(label="No content extracted from PDFs", state="error")
                return False

    def ask(self, query):
        """Ask a question and get an answer based on the PDFs with blockchain logging."""
        if not self.vector_store:
            return "Please upload and process PDF files first."
            
        try:
            # Custom prompt
            prompt_template = """
            You are an AI assistant that provides accurate information based on PDF documents.
            
            Use the following context to answer the question. Be detailed and precise in your answer.
            If the answer is not in the context, say "I don't have enough information to answer this question."
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            # Start timing the query
            query_start_time = time.time()
            
            # Create QA chain
            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True
            )
            
            # Get answer
            with st.status("Searching documents and generating answer..."):
                response = qa({"query": query})
                
            answer = response["result"]
            source_docs = response["source_documents"]
            
            # Calculate query time
            query_time = time.time() - query_start_time
            
            # Format sources
            sources = []
            for i, doc in enumerate(source_docs):
                # Extract blockchain verification info if available
                blockchain_info = None
                if "blockchain" in doc.metadata:
                    blockchain_info = {
                        "verified": doc.metadata["blockchain"]["verified"],
                        "document_id": doc.metadata["blockchain"]["document_id"],
                        "tx_hash": doc.metadata["blockchain"]["tx_hash"]
                    }
                
                sources.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "blockchain": blockchain_info
                })
            
            # Log query to blockchain if enabled
            blockchain_log = None
            if self.use_blockchain and self.blockchain:
                try:
                    with st.status("Logging query to blockchain..."):
                        log_result = self.blockchain.log_query(query, answer)
                        
                        if log_result["status"] == 1:
                            blockchain_log = {
                                "logged": True,
                                "query_id": log_result["query_id"],
                                "tx_hash": log_result["tx_hash"],
                                "block_number": log_result["block_number"]
                            }
                except Exception as e:
                    st.error(f"Error logging to blockchain: {str(e)}")
                
            return {
                "answer": answer,
                "sources": sources,
                "query_time": query_time,
                "blockchain_log": blockchain_log
            }
                
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return f"Error: {str(e)}"

    def get_performance_metrics(self):
        """Return performance metrics for the RAG system."""
        if not self.processing_times:
            return None
            
        return {
            "documents_processed": self.documents_processed,
            "index_building_time": self.processing_times.get("index_building", 0),
            "total_processing_time": self.processing_times.get("total_time", 0),
            "memory_used_gb": self.processing_times.get("memory_used_gb", 0),
            "device": self.device,
            "embedding_model": self.embedding_model_name,
            "blockchain_enabled": self.use_blockchain
        }


# Helper function to initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None

# Helper function to clean up temporary files
def cleanup_temp_files():
    """Clean up temporary files when application exits."""
    if st.session_state.get('temp_dir') and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
            print(f"Cleaned up temporary directory: {st.session_state.temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")


# Streamlit UI
def main():
    st.set_page_config(page_title="Blockchain-Enabled RAG System", layout="wide")
    
    st.title("🚀 GPU-Accelerated PDF Question Answering with Blockchain Verification")
    st.markdown("Upload PDFs, verify them on blockchain, and ask questions with audit log")
    
    # Initialize session state
    initialize_session_state()
    
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
            options=["llama3.2:latest", "llama3:latest", "mistral:latest"],
            index=0
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ],
            index=1  # all-MiniLM-L6-v2 is smaller and faster
        )
        
        use_gpu = st.checkbox("Use GPU Acceleration", value=gpu_available)
        
        # Blockchain configuration
        st.header("🔗 Blockchain Configuration")
        use_blockchain = st.checkbox("Enable Blockchain Verification", value=False)
        
        if use_blockchain:
            blockchain_url = st.text_input("Blockchain URL", value="http://127.0.0.1:7545")
            contract_address = st.text_input("Contract Address", 
                                            value="0x0000000000000000000000000000000000000000")
            private_key = st.text_input("Private Key (without 0x prefix)", 
                                      value="", type="password")
            
            # Display warning about private key security
            st.warning("⚠️ Never share your private key. For production, use secure key management.")
            
            if not contract_address or contract_address == "0x0000000000000000000000000000000000000000":
                st.error("Please deploy the contract and enter its address")
        
        # Advanced options
        with st.expander("Advanced Options"):
            chunk_size = st.slider("Chunk Size", 100, 2000, 1000)
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
        
        # Initialize button
        if st.button("Initialize System"):
            with st.spinner("Initializing RAG system..."):
                if use_blockchain and (not contract_address or not private_key):
                    st.error("Contract address and private key are required for blockchain integration")
                else:
                    st.session_state.rag = BlockchainEnabledRAG(
                        llm_model_name=llm_model,
                        embedding_model_name=embedding_model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        use_gpu=use_gpu and gpu_available,
                        use_blockchain=use_blockchain,
                        blockchain_url=blockchain_url if use_blockchain else None,
                        contract_address=contract_address if use_blockchain else None,
                        private_key=private_key if use_blockchain else None
                    )
                    
                    st.success(f"System initialized with {embedding_model} on {st.session_state.rag.device}")
                    if use_blockchain and st.session_state.rag.blockchain:
                        st.success("Blockchain verification enabled")
        
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader("Select PDFs", type="pdf", accept_multiple_files=True)
        
        if uploaded_files and st.button("Process PDFs"):
            if not st.session_state.rag:
                with st.spinner("Initializing RAG system..."):
                    st.session_state.rag = BlockchainEnabledRAG(
                        llm_model_name=llm_model,
                        embedding_model_name=embedding_model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        use_gpu=use_gpu and gpu_available,
                        use_blockchain=use_blockchain,
                        blockchain_url=blockchain_url if use_blockchain else None,
                        contract_address=contract_address if use_blockchain else None,
                        private_key=private_key if use_blockchain else None
                    )
            
            success = st.session_state.rag.process_pdfs(uploaded_files)
            if success:
                metrics = st.session_state.rag.get_performance_metrics()
                if metrics:
                    st.success("PDFs processed successfully!")
                    with st.expander("💹 Performance Metrics"):
                        st.markdown(f"**Documents processed:** {metrics['documents_processed']} chunks")
                        st.markdown(f"**Index building time:** {metrics['index_building_time']:.2f} seconds")
                        st.markdown(f"**Total processing time:** {metrics['total_processing_time']:.2f} seconds")
                        st.markdown(f"**Memory used:** {metrics['memory_used_gb']:.2f} GB")
                        st.markdown(f"**Device used:** {metrics['device']}")
                        st.markdown(f"**Blockchain verification:** {'Enabled' if metrics['blockchain_enabled'] else 'Disabled'}")
    
    # Blockchain verification info
    if st.session_state.rag and st.session_state.rag.use_blockchain:
        st.info("🔗 Blockchain verification is enabled. Documents are cryptographically verified and queries are logged with immutable audit trail.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if isinstance(message["content"], dict):
                    st.markdown(message["content"]["answer"])
                    
                    if "query_time" in message["content"]:
                        st.caption(f"Response time: {message['content']['query_time']:.2f} seconds")
                    
                    # Display blockchain log if available
                    if "blockchain_log" in message["content"] and message["content"]["blockchain_log"]:
                        blockchain_log = message["content"]["blockchain_log"]
                        st.success(f"✅ Query logged on blockchain | Transaction: {blockchain_log['tx_hash'][:10]}...")
                    
                    # Display sources in expander
                    if "sources" in message["content"] and message["content"]["sources"]:
                        with st.expander("📄 View Sources"):
                            for i, source in enumerate(message["content"]["sources"]):
                                st.markdown(f"**Source {i+1}: {source['source']}**")
                                
                                # Show blockchain verification if available
                                if source.get("blockchain"):
                                    st.success(f"✅ Verified on blockchain | TX: {source['blockchain']['tx_hash'][:10]}...")
                                    
                                st.text(source["content"])
                                st.divider()
                else:
                    st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your PDFs..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if system is initialized
        if not st.session_state.rag:
            with st.chat_message("assistant"):
                message = "Please initialize the system and process PDFs first."
                st.markdown(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
        
        # Get response if vector store is ready
        elif st.session_state.rag.vector_store:
            with st.chat_message("assistant"):
                response = st.session_state.rag.ask(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                if isinstance(response, dict):
                    st.markdown(response["answer"])
                    
                    if "query_time" in response:
                        st.caption(f"Response time: {response['query_time']:.2f} seconds")
                    
                    # Display blockchain log if available
                    if "blockchain_log" in response and response["blockchain_log"]:
                        blockchain_log = response["blockchain_log"]
                        st.success(f"✅ Query logged on blockchain | Transaction: {blockchain_log['tx_hash'][:10]}...")
                    
                    # Display sources in expander
                    if "sources" in response and response["sources"]:
                        with st.expander("📄 View Sources"):
                            for i, source in enumerate(response["sources"]):
                                st.markdown(f"**Source {i+1}: {source['source']}**")
                                
                                # Show blockchain verification if available
                                if source.get("blockchain"):
                                    st.success(f"✅ Verified on blockchain | TX: {source['blockchain']['tx_hash'][:10]}...")
                                    
                                st.text(source["content"])
                                st.divider()
                else:
                    st.markdown(response)
        else:
            with st.chat_message("assistant"):
                message = "Please upload and process PDF files first."
                st.markdown(message)
                st.session_state.messages.append({"role": "assistant", "content": message})


# Main entry point
if __name__ == "__main__":
    # Register cleanup function
    atexit.register(cleanup_temp_files)
    
    main()