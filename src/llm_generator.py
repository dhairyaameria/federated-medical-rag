from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict
import os

class MedicalLLMGenerator:
    """Generate answers using LLM with retrieved context"""
    
    def __init__(self, provider: str = "anthropic", model_name: str = "claude-3-5-sonnet-20241022", temperature: float = 0.1):
        """
        Initialize LLM Generator with Anthropic Claude
        
        Args:
            provider: Should be "anthropic"
            model_name: Claude model identifier
            temperature: Sampling temperature
        """
        # Load API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError(
                "\n❌ Anthropic API key not found!\n\n"
                "Please set your API key:\n"
                "  Option 1: export ANTHROPIC_API_KEY='sk-ant-...'\n"
                "  Option 2: Add to .env file: ANTHROPIC_API_KEY=sk-ant-...\n\n"
                "Get your key at: https://console.anthropic.com/settings/keys"
            )
        
        print(f"Initializing Claude {model_name}...")
        
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            max_tokens=1024
        )
        
        print(f"✓ Claude {model_name} ready!")
        
        self.system_prompt = """You are a medical AI assistant helping healthcare professionals.
Your task is to answer medical questions based ONLY on the provided context documents.

Guidelines:
1. Base your answer strictly on the provided context
2. Cite the source documents using [1], [2], etc.
3. If the context doesn't contain enough information, say so clearly
4. Be precise and use medical terminology appropriately
5. Do not make up information or hallucinate
6. If asked about treatments, always recommend consulting with healthcare providers"""
    
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
            ],
            'model': self.llm.model
        }
        
        return result