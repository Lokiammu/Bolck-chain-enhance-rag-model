# Blockchain and RAG Project

This project consists of two main components:
1. **Blockchain Module** (Located in `blockchain/`)
2. **RAG (Retrieval-Augmented Generation) Module** (Located in `rag/`)

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.12+
- Pip
- Streamlit (for the RAG module)
- Web3.py (for blockchain integration)
- Remix IDE (for smart contract execution)
- Ganache (for private blockchain API key management)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Lokiammu/Bolck-chain-enhance-rag-model
   cd project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### **Blockchain Module**
- Deploy the smart contract using Remix IDE.
- Use Ganache for private API key management and local blockchain testing.
- Once deployed, navigate to the blockchain directory and execute:
   ```bash
   streamlit run blockchain/main.py
   ```
This script handles blockchain-related functionalities.

### **RAG Module**
To launch the RAG module with Streamlit, run:
   ```bash
   streamlit run rag/app.py
   ```
This will start a web interface for interacting with the RAG system.

## Project Structure
```
📂 project_root
├── blockchain
│   ├── blockchain_utils.py
│   ├── main.py
│   ├── RAGDocumentVerifier.sol
│   └── __pycache__/
├── rag
│   ├── app.py
│   ├── auth.py
│   ├── chat.py
│   ├── database.py
│   ├── document_viewer.py
│   ├── notebooks/
│   ├── rag.py
│   ├── settings.py
│   ├── utils.py
│   └── __pycache__/
├── requirements.txt
└── README.md
```

## Contributing
Feel free to open issues or submit pull requests to improve the project.

---
**Author:** Lokesh Narne
For questions, reach out via email or GitHub issues.

