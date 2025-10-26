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