// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract RAGDocumentVerifier {
    // Mapping from document ID to document hash
    mapping(string => string) public documentHashes;
    
    // Mapping from query ID to query hash
    mapping(string => string) public queryHashes;
    
    // Events
    event DocumentVerified(
        address indexed user,
        string indexed documentId,
        string documentHash,
        uint256 timestamp
    );
    
    event QueryLogged(
        address indexed user,
        string indexed queryId,
        string queryHash, 
        uint256 timestamp
    );
    
    constructor() {                                                           
        // Constructor
    }
    
    /**
     * @dev Verify a document by storing its hash
     * @param documentId Unique identifier for the document
     * @param documentHash Hash of the document content
     */
    function verifyDocument(string memory documentId, string memory documentHash) public {
        documentHashes[documentId] = documentHash;
        
        emit DocumentVerified(
            msg.sender,
            documentId,
            documentHash,
            block.timestamp
        );
    }
    
    /**
     * @dev Log a query and its hash
     * @param queryId Unique identifier for the query
     * @param queryHash Hash of the query data (including query text and answer)
     */
    function logQuery(string memory queryId, string memory queryHash) public {
        queryHashes[queryId] = queryHash;
        
        emit QueryLogged(
            msg.sender,
            queryId,
            queryHash,
            block.timestamp
        );
    }
    
    /**
     * @dev Get document hash by ID
     * @param documentId Document ID to look up
     * @return Document hash
     */
    function getDocumentHash(string memory documentId) public view returns (string memory) {
        return documentHashes[documentId];
    }
    
    /**
     * @dev Get query information by ID
     * @param queryId Query ID to look up
     * @return Query hash
     */
    function getQueryInfo(string memory queryId) public view returns (string memory) {
        return queryHashes[queryId];
    }
}