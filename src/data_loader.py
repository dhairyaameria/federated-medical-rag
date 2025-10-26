from datasets import load_dataset, load_from_disk
from typing import List, Dict
import json
import os

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