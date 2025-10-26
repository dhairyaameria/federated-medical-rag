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