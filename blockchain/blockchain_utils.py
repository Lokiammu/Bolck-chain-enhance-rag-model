# blockchain_utils.py
import hashlib
import json
import os
from web3 import Web3
import streamlit as st
import time

class BlockchainManager:
    def __init__(self, 
                 blockchain_url="http://localhost:8545", 
                 chain_id=1337,
                 contract_address=None,
                 private_key=None):
        """
        Initialize blockchain connection and contract interfaces.
        
        Args:
            blockchain_url: URL of the blockchain node (default: local Ganache)
            chain_id: Chain ID for the blockchain network
            contract_address: Address of the deployed RAG Document Verifier contract
            private_key: Private key for signing transactions (without 0x prefix)
        """
        self.w3 = Web3(Web3.HTTPProvider(blockchain_url))
        
        # Check connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to blockchain at {blockchain_url}")
            
        self.chain_id = chain_id
        self.contract_address = contract_address
        
        # Add 0x prefix to private key if not present
        if private_key and not private_key.startswith('0x'):
            private_key = '0x' + private_key
        self.private_key = private_key
        
        # Load account from private key if provided
        self.account = None
        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
        
        # Load contract ABI
        self.contract = None
        if contract_address:
            self.load_contract()
    
    def load_contract(self):
        """Load the RAGDocumentVerifier contract interface."""
        # ABI for the RAG Document Verifier contract
        abi = [
            {
                "inputs": [],
                "stateMutability": "nonpayable",
                "type": "constructor"
            },
            {
                "anonymous": False,
                "inputs": [
                    {
                        "indexed": True,
                        "internalType": "address",
                        "name": "user",
                        "type": "address"
                    },
                    {
                        "indexed": True,
                        "internalType": "string",
                        "name": "documentId",
                        "type": "string"
                    },
                    {
                        "indexed": False,
                        "internalType": "string",
                        "name": "documentHash",
                        "type": "string"
                    },
                    {
                        "indexed": False,
                        "internalType": "uint256",
                        "name": "timestamp",
                        "type": "uint256"
                    }
                ],
                "name": "DocumentVerified",
                "type": "event"
            },
            {
                "anonymous": False,
                "inputs": [
                    {
                        "indexed": True,
                        "internalType": "address",
                        "name": "user",
                        "type": "address"
                    },
                    {
                        "indexed": True,
                        "internalType": "string",
                        "name": "queryId",
                        "type": "string"
                    },
                    {
                        "indexed": False,
                        "internalType": "string",
                        "name": "queryHash",
                        "type": "string"
                    },
                    {
                        "indexed": False,
                        "internalType": "uint256",
                        "name": "timestamp",
                        "type": "uint256"
                    }
                ],
                "name": "QueryLogged",
                "type": "event"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "name": "documentHashes",
                "outputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "documentId",
                        "type": "string"
                    }
                ],
                "name": "getDocumentHash",
                "outputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "queryId",
                        "type": "string"
                    }
                ],
                "name": "getQueryInfo",
                "outputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "queryId",
                        "type": "string"
                    },
                    {
                        "internalType": "string",
                        "name": "queryHash",
                        "type": "string"
                    }
                ],
                "name": "logQuery",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "name": "queryHashes",
                "outputs": [
                    {
                        "internalType": "string",
                        "name": "",
                        "type": "string"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "documentId",
                        "type": "string"
                    },
                    {
                        "internalType": "string",
                        "name": "documentHash",
                        "type": "string"
                    }
                ],
                "name": "verifyDocument",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        # Create contract instance
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=abi)
    
    def compute_file_hash(self, file_path):
        """
        Compute the SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Hexadecimal hash of the file
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read and update hash in chunks for memory efficiency
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
                
        return sha256_hash.hexdigest()
    
    def verify_document(self, document_id, file_path):
        """
        Verify a document by storing its hash on the blockchain.
        
        Args:
            document_id: Unique identifier for the document
            file_path: Path to the document file
            
        Returns:
            dict: Transaction receipt
        """
        if not self.contract or not self.account:
            raise ValueError("Contract address or private key not set")
            
        # Compute document hash
        document_hash = self.compute_file_hash(file_path)
        
        # Build transaction
        tx = self.contract.functions.verifyDocument(
            document_id,
            document_hash
        ).build_transaction({
            'chainId': self.chain_id,
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
        })
        
        # Sign and send transaction - FIXED for Web3.py compatibility
        try:
            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
            
            # Get the raw transaction - handle different Web3.py versions
            if hasattr(signed_tx, 'rawTransaction'):
                raw_tx = signed_tx.rawTransaction
            elif hasattr(signed_tx, 'raw_transaction'):
                raw_tx = signed_tx.raw_transaction
            else:
                # Direct access to __dict__ as a fallback
                raw_tx = list(signed_tx.__dict__.values())[0]
                
            # Send raw transaction    
            tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
            
            # Wait for transaction receipt
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                'tx_hash': tx_hash.hex(),
                'document_id': document_id,
                'document_hash': document_hash,
                'block_number': tx_receipt['blockNumber'],
                'status': tx_receipt['status']
            }
        except Exception as e:
            st.error(f"Transaction error: {str(e)}")
            # Print additional debug info
            print(f"Transaction dictionary: {tx}")
            print(f"Account address: {self.account.address}")
            print(f"Private key (first 4 chars): {self.private_key[:6]}...")
            raise
    
    def check_document_verified(self, document_id):
        """
        Check if a document has already been verified on the blockchain.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            bool: True if document is verified, False otherwise
        """
        if not self.contract:
            raise ValueError("Contract address not set")
            
        stored_hash = self.contract.functions.getDocumentHash(document_id).call()
        return stored_hash != ""
    
    def log_query(self, query_text, answer_text):
        """
        Log a query and its answer on the blockchain.
        
        Args:
            query_text: The user's query
            answer_text: The system's answer
            
        Returns:
            dict: Transaction receipt
        """
        if not self.contract or not self.account:
            raise ValueError("Contract address or private key not set")
            
        # Create a unique query ID using timestamp
        query_id = f"query_{int(time.time())}"
        
        # Create a JSON object with query and answer
        query_data = {
            "query": query_text,
            "answer": answer_text,
            "timestamp": int(time.time())
        }
        
        # Hash the query data
        query_hash = hashlib.sha256(json.dumps(query_data).encode()).hexdigest()
        
        # Build transaction
        tx = self.contract.functions.logQuery(
            query_id,
            query_hash
        ).build_transaction({
            'chainId': self.chain_id,
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
        })
        
        # Sign and send transaction - FIXED for Web3.py compatibility
        try:
            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
            
            # Get the raw transaction - handle different Web3.py versions
            if hasattr(signed_tx, 'rawTransaction'):
                raw_tx = signed_tx.rawTransaction
            elif hasattr(signed_tx, 'raw_transaction'):
                raw_tx = signed_tx.raw_transaction
            else:
                # Direct access to __dict__ as a fallback
                raw_tx = list(signed_tx.__dict__.values())[0]
                
            # Send raw transaction    
            tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
            
            # Wait for transaction receipt
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                'tx_hash': tx_hash.hex(),
                'query_id': query_id,
                'query_hash': query_hash,
                'block_number': tx_receipt['blockNumber'],
                'status': tx_receipt['status']
            }
        except Exception as e:
            st.error(f"Transaction error: {str(e)}")
            # Print additional debug info
            print(f"Transaction dictionary: {tx}")
            print(f"Account address: {self.account.address}")
            print(f"Private key (first 4 chars): {self.private_key[:6]}...")
            raise