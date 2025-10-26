# Federated Medical RAG - Starter Code Template

## Installation Guide

### Prerequisites
```bash
# Python 3.9+
python --version

# Create virtual environment
python -m venv fedrag_env
source fedrag_env/bin/activate  # On Windows: fedrag_env\Scripts\activate
```

### Install Dependencies
```bash
# Core dependencies
pip install flwr[simulation]==1.11.1
pip install fed-rag[qdrant,llamaindex]==0.0.27
pip install torch torchvision torchaudio
pip install transformers sentence-transformers
pip install llama-index llama-index-vector-stores-qdrant
pip install langchain langchain-openai langchain-community
pip install qdrant-client
pip install datasets
pip install openai anthropic

# For development
pip install jupyter notebook
pip install matplotlib seaborn
pip install pytest black flake8
```

### Download PubMedQA Dataset
```python
from datasets import load_dataset

# Load PubMedQA dataset
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

# Save locally
dataset.save_to_disk("./data/pubmedqa")
```

---

## Project Structure

```
federated-medical-rag/
├── data/
│   ├── pubmedqa/              # Downloaded dataset
│   ├── hospital_a/            # Client 1 data
│   ├── hospital_b/            # Client 2 data
│   └── research_center/       # Client 3 data
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # Text preprocessing
│   ├── embeddings.py          # Embedding models
│   ├── vector_store.py        # Vector database operations
│   ├── retrieval.py           # Retrieval logic
│   ├── server.py              # Flower server
│   ├── client.py              # Flower client
│   └── llm_generator.py       # LLM response generation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_rag.ipynb
│   └── 03_federated_rag.ipynb
├── tests/
│   └── test_retrieval.py
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 1. Configuration (src/config.py)

```python
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class ModelConfig:
    """Model configuration"""
    embedding_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    llm_model: str = "gpt-4"
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.1
    max_tokens: int = 1024

@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    top_k: int = 10
    similarity_threshold: float = 0.7
    chunk_size: int = 512
    chunk_overlap: int = 50
    rerank: bool = True

@dataclass
class FlowerConfig:
    """Flower federated learning configuration"""
    num_clients: int = 3
    num_rounds: int = 1  # For RAG, typically 1 round per query
    server_address: str = "localhost:8080"
    k_rrf: int = 60  # Reciprocal Rank Fusion parameter

@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    store_type: str = "qdrant"
    collection_name: str = "medical_documents"
    host: str = "localhost"
    port: int = 6333
    dimension: int = 768  # BioBERT dimension

class Config:
    """Main configuration class"""
    def __init__(self):
        self.model = ModelConfig()
        self.retrieval = RetrievalConfig()
        self.flower = FlowerConfig()
        self.vector_store = VectorStoreConfig()
        
        # Set API keys from environment
        self.model.llm_api_key = os.getenv("OPENAI_API_KEY")

# Global config instance
config = Config()
```

---

## 2. Data Loader (src/data_loader.py)

```python
from datasets import load_dataset, load_from_disk
from typing import List, Dict
import json

class PubMedQALoader:
    """Load and split PubMedQA dataset for federated clients"""
    
    def __init__(self, data_path: str = "./data/pubmedqa"):
        self.data_path = data_path
        self.dataset = None
    
    def load_dataset(self):
        """Load PubMedQA dataset"""
        try:
            self.dataset = load_from_disk(self.data_path)
        except:
            print("Downloading PubMedQA dataset...")
            self.dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
            self.dataset.save_to_disk(self.data_path)
        return self.dataset
    
    def split_by_specialty(self, num_clients: int = 3) -> List[Dict]:
        """Split dataset by medical specialty for different clients"""
        if self.dataset is None:
            self.load_dataset()
        
        # Get training data
        train_data = self.dataset['train']
        
        # Simple split for demo (in production, split by topic/specialty)
        split_size = len(train_data) // num_clients
        client_datasets = []
        
        for i in range(num_clients):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < num_clients - 1 else len(train_data)
            
            client_data = {
                'documents': [],
                'metadata': []
            }
            
            for idx in range(start_idx, end_idx):
                example = train_data[idx]
                
                # Combine context and long_answer as document
                document = f"Question: {example['question']}\n\n"
                document += f"Context: {example['context']}\n\n"
                document += f"Answer: {example['long_answer']}"
                
                client_data['documents'].append(document)
                client_data['metadata'].append({
                    'question': example['question'],
                    'final_decision': example['final_decision'],
                    'pubid': example.get('pubid', ''),
                    'client_id': i
                })
            
            client_datasets.append(client_data)
            print(f"Client {i}: {len(client_data['documents'])} documents")
        
        return client_datasets
    
    def save_client_data(self, client_datasets: List[Dict], base_path: str = "./data"):
        """Save split datasets for each client"""
        client_names = ['hospital_a', 'hospital_b', 'research_center']
        
        for i, client_data in enumerate(client_datasets):
            client_path = f"{base_path}/{client_names[i]}"
            os.makedirs(client_path, exist_ok=True)
            
            # Save documents
            with open(f"{client_path}/documents.json", 'w') as f:
                json.dump(client_data, f, indent=2)
            
            print(f"Saved data for {client_names[i]}")
```

---

## 3. Preprocessing (src/preprocessing.py)

```python
from typing import List, Dict
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MedicalTextPreprocessor:
    """Preprocess medical text for RAG"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep medical notation
        text = re.sub(r'[^\w\s\.\,\-\(\)\:\;\%\/]', '', text)
        return text.strip()
    
    def chunk_documents(self, documents: List[str]) -> List[Dict]:
        """Split documents into chunks"""
        chunks = []
        for doc_id, doc in enumerate(documents):
            cleaned_doc = self.clean_text(doc)
            doc_chunks = self.text_splitter.split_text(cleaned_doc)
            
            for chunk_id, chunk in enumerate(doc_chunks):
                chunks.append({
                    'text': chunk,
                    'doc_id': doc_id,
                    'chunk_id': chunk_id
                })
        
        return chunks
    
    def extract_medical_entities(self, text: str) -> List[str]:
        """Simple medical entity extraction (can be enhanced with scispacy)"""
        # Placeholder - in production use scispacy or BioBERT NER
        # For now, extract capitalized medical terms
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(entities))
```

---

## 4. Vector Store (src/vector_store.py)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import numpy as np

class QdrantVectorStore:
    """Qdrant vector database manager"""
    
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 collection_name: str = "medical_docs"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
    
    def create_collection(self, dimension: int = 768):
        """Create vector collection"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        except Exception as e:
            print(f"Collection might already exist: {e}")
    
    def add_documents(self, documents: List[str], 
                     embeddings: np.ndarray, 
                     metadata: List[Dict]):
        """Add documents with embeddings to vector store"""
        points = []
        for idx, (doc, emb, meta) in enumerate(zip(documents, embeddings, metadata)):
            point = PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload={
                    'text': doc,
                    **meta
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Added {len(points)} documents to {self.collection_name}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Search for similar documents"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        return [
            {
                'text': hit.payload['text'],
                'score': hit.score,
                'metadata': {k: v for k, v in hit.payload.items() if k != 'text'}
            }
            for hit in results
        ]
```

---

## 5. Embeddings (src/embeddings.py)

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class BioBERTEmbedder:
    """BioBERT embedding model for medical text"""
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query"""
        return self.model.encode([query], convert_to_numpy=True)[0]
```

---

## 6. Flower Client (src/client.py)

```python
import flwr as fl
from typing import Dict, List
import json
from src.embeddings import BioBERTEmbedder
from src.vector_store import QdrantVectorStore
from src.preprocessing import MedicalTextPreprocessor
from src.config import config

class MedicalRAGClient(fl.client.NumPyClient):
    """Flower client for federated medical RAG"""
    
    def __init__(self, client_id: int, data_path: str):
        self.client_id = client_id
        self.data_path = data_path
        
        # Initialize components
        self.embedder = BioBERTEmbedder(config.model.embedding_model)
        self.preprocessor = MedicalTextPreprocessor(
            chunk_size=config.retrieval.chunk_size,
            chunk_overlap=config.retrieval.chunk_overlap
        )
        self.vector_store = QdrantVectorStore(
            host=config.vector_store.host,
            port=config.vector_store.port,
            collection_name=f"{config.vector_store.collection_name}_client_{client_id}"
        )
        
        # Load and index local documents
        self._load_and_index_documents()
    
    def _load_and_index_documents(self):
        """Load and index client's local documents"""
        print(f"Client {self.client_id}: Loading documents...")
        
        # Load documents
        with open(f"{self.data_path}/documents.json", 'r') as f:
            data = json.load(f)
        
        documents = data['documents']
        metadata = data['metadata']
        
        # Preprocess
        chunks = self.preprocessor.chunk_documents(documents)
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"Client {self.client_id}: Generating embeddings...")
        embeddings = self.embedder.encode(chunk_texts)
        
        # Create vector store collection
        self.vector_store.create_collection(dimension=self.embedder.dimension)
        
        # Add to vector store
        chunk_metadata = [
            {**metadata[chunk['doc_id']], 'chunk_id': chunk['chunk_id']}
            for chunk in chunks
        ]
        self.vector_store.add_documents(chunk_texts, embeddings, chunk_metadata)
        
        print(f"Client {self.client_id}: Indexed {len(chunk_texts)} chunks")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        # Encode query
        query_embedding = self.embedder.encode_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return results
```

---

## 7. Flower Server (src/server.py)

```python
import flwr as fl
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

class FederatedRAGServer:
    """Flower server for federated RAG aggregation"""
    
    def __init__(self, num_clients: int = 3, k_rrf: int = 60):
        self.num_clients = num_clients
        self.k_rrf = k_rrf
    
    def reciprocal_rank_fusion(self, 
                               client_results: List[List[Dict]], 
                               k: int = 60) -> List[Dict]:
        """
        Aggregate results from multiple clients using Reciprocal Rank Fusion
        
        Args:
            client_results: List of result lists from each client
            k: RRF constant (default 60)
        
        Returns:
            Merged and ranked results
        """
        doc_scores = defaultdict(float)
        doc_data = {}
        
        # Calculate RRF scores
        for client_results_list in client_results:
            for rank, doc in enumerate(client_results_list, start=1):
                doc_id = doc['text'][:100]  # Use text prefix as ID
                rrf_score = 1.0 / (k + rank)
                doc_scores[doc_id] += rrf_score
                
                if doc_id not in doc_data:
                    doc_data[doc_id] = doc
        
        # Sort by aggregated scores
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top documents with scores
        merged_results = [
            {
                **doc_data[doc_id],
                'rrf_score': score
            }
            for doc_id, score in sorted_docs
        ]
        
        return merged_results
    
    def aggregate_results(self, query: str, client_results: List[List[Dict]]) -> List[Dict]:
        """Aggregate results from all clients"""
        print(f"\nAggregating results from {len(client_results)} clients...")
        
        # Use RRF to merge results
        merged_results = self.reciprocal_rank_fusion(client_results, k=self.k_rrf)
        
        print(f"Merged into {len(merged_results)} unique documents")
        return merged_results
```

---

## 8. LLM Generator (src/llm_generator.py)

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from typing import List, Dict
import os

class MedicalLLMGenerator:
    """Generate answers using LLM with retrieved context"""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.1):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.system_prompt = """You are a medical AI assistant helping healthcare professionals.
        Your task is to answer medical questions based ONLY on the provided context documents.
        
        Guidelines:
        1. Base your answer strictly on the provided context
        2. Cite the source documents using [1], [2], etc.
        3. If the context doesn't contain enough information, say so
        4. Be precise and use medical terminology appropriately
        5. Do not make up information or hallucinate
        """
    
    def generate_answer(self, query: str, context_docs: List[Dict], top_k: int = 5) -> Dict:
        """Generate answer with citations"""
        
        # Prepare context
        context_texts = []
        for i, doc in enumerate(context_docs[:top_k], start=1):
            context_texts.append(f"[{i}] {doc['text']}\n")
        
        context = "\n".join(context_texts)
        
        # Create prompt
        user_prompt = f"""Question: {query}

Context Documents:
{context}

Please provide a comprehensive answer to the question based on the context above.
Include citations [1], [2], etc. to reference the source documents."""
        
        # Generate response
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Format result
        result = {
            'query': query,
            'answer': response.content,
            'sources': [
                {
                    'id': i,
                    'text': doc['text'][:200] + "...",
                    'score': doc.get('rrf_score', doc.get('score', 0)),
                    'metadata': doc.get('metadata', {})
                }
                for i, doc in enumerate(context_docs[:top_k], start=1)
            ]
        }
        
        return result
```

---

## 9. Main Execution Script

```python
# main.py
from src.config import config
from src.data_loader import PubMedQALoader
from src.client import MedicalRAGClient
from src.server import FederatedRAGServer
from src.llm_generator import MedicalLLMGenerator
import argparse

def setup_clients():
    """Setup and prepare client data"""
    print("=== Setting up Federated Clients ===\n")
    
    # Load and split dataset
    loader = PubMedQALoader()
    loader.load_dataset()
    client_datasets = loader.split_by_specialty(num_clients=config.flower.num_clients)
    loader.save_client_data(client_datasets)
    
    print("\nClient data prepared successfully!")

def run_federated_query(query: str):
    """Run a federated query across all clients"""
    print(f"\n=== Federated Query: {query} ===\n")
    
    # Initialize clients
    clients = []
    client_paths = ['./data/hospital_a', './data/hospital_b', './data/research_center']
    
    for i, path in enumerate(client_paths):
        print(f"\nInitializing Client {i}...")
        client = MedicalRAGClient(client_id=i, data_path=path)
        clients.append(client)
    
    # Execute retrieval on each client
    print(f"\n=== Retrieving from {len(clients)} clients ===")
    client_results = []
    for i, client in enumerate(clients):
        print(f"\nClient {i} retrieving...")
        results = client.retrieve(query, top_k=config.retrieval.top_k)
        client_results.append(results)
        print(f"Client {i}: Found {len(results)} documents")
    
    # Aggregate results on server
    server = FederatedRAGServer(
        num_clients=config.flower.num_clients,
        k_rrf=config.flower.k_rrf
    )
    merged_results = server.aggregate_results(query, client_results)
    
    # Generate answer using LLM
    print("\n=== Generating Answer with LLM ===")
    generator = MedicalLLMGenerator(
        model_name=config.model.llm_model,
        temperature=config.model.llm_temperature
    )
    result = generator.generate_answer(query, merged_results, top_k=5)
    
    # Display result
    print("\n" + "="*80)
    print(f"QUERY: {result['query']}")
    print("="*80)
    print(f"\nANSWER:\n{result['answer']}")
    print("\n" + "="*80)
    print("SOURCES:")
    for source in result['sources']:
        print(f"\n[{source['id']}] Score: {source['score']:.4f}")
        print(f"    {source['text']}")
        print(f"    Metadata: {source['metadata']}")
    print("="*80)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Federated Medical RAG System')
    parser.add_argument('--setup', action='store_true', help='Setup client data')
    parser.add_argument('--query', type=str, help='Query to run')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_clients()
    elif args.query:
        run_federated_query(args.query)
    else:
        # Demo queries
        demo_queries = [
            "What are the side effects of Metformin?",
            "How effective is aspirin for cardiovascular disease prevention?",
            "What are the latest treatments for Type 2 Diabetes?"
        ]
        
        print("\n=== Running Demo Queries ===\n")
        for query in demo_queries:
            run_federated_query(query)
            print("\n\n")

if __name__ == "__main__":
    main()
```

---

## Usage

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup clients (split and index data)
python main.py --setup

# Start Qdrant (in separate terminal)
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Run Queries
```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Run single query
python main.py --query "What are the side effects of Metformin?"

# Run demo queries
python main.py
```

### 3. Docker Deployment
```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f
```

---

## Next Steps

1. **Enhance Retrieval**: Add reranking, hybrid search (BM25 + Dense)
2. **Add Authentication**: Implement mTLS between clients
3. **Monitoring**: Add Prometheus metrics and Grafana dashboards
4. **Web Interface**: Build Gradio/Streamlit UI
5. **Fine-tuning**: Use FedRAG to fine-tune retrieval models
6. **Multi-modal**: Add support for medical images and tables

---

## Resources

- [Flower Documentation](https://flower.ai/docs/)
- [FedRAG GitHub](https://github.com/VectorInstitute/fed-rag)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [PubMedQA Dataset](https://pubmedqa.github.io/)
- [BioBERT Paper](https://arxiv.org/abs/1901.08746)
