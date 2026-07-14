import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerBase, PreTrainedModel
import threading
import numpy as np
from typing import Any, Optional

class ModernBertEmbedderSingleton:
    _instance: Optional["ModernBertEmbedderSingleton"] = None
    _lock: threading.Lock = threading.Lock()
    tokenizer: PreTrainedTokenizerBase
    model: Any

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModernBertEmbedderSingleton, cls).__new__(cls)
                cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        # Local model footprint verification path
        self.model_name = "answerdotai/ModernBERT-base"
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load local configurations
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer is None:
            raise RuntimeError("Failed to load tokenizer")
        self.tokenizer = tokenizer

        model = AutoModel.from_pretrained(self.model_name)
        if model is None:
            raise RuntimeError("Failed to load model")
        self.model = model

        self.model.to(self.device)
        self.model.eval()  # Freeze graph gradients

    def compute_embedding(self, title: str, description: str) -> list:
        # Preprocessing: Clean strings and construct canonical structural form
        combined_text = f"Title: {title.strip()} | Description: {description.strip()}"
        
        # Tokenization & Target Bound Enforcement
        inputs = self.tokenizer(
            combined_text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Perform Mean Pooling over attention weight configurations
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # L2 Normalization sequence to enable optimized Cosine Metric comparisons
            norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return norm_embeddings.cpu().squeeze().numpy().tolist()
