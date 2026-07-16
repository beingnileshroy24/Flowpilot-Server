import numpy as np
import logging
import json
import re
from typing import List, Dict, Any, Optional
import uuid

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, lancedb_manager=None, embedder=None):
        from app.ai.storage.lancedb_client import LanceDBManager
        from app.ai.core.embedder import ModernBertEmbedderSingleton
        self.lancedb_manager = lancedb_manager or LanceDBManager()
        self.embedder = embedder or ModernBertEmbedderSingleton()

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r'\b\w+\b', text.lower())

    def _score_keyword_match(self, query_tokens: List[str], text_tokens: List[str]) -> float:
        if not query_tokens or not text_tokens:
            return 0.0
        query_set = set(query_tokens)
        text_set = set(text_tokens)
        overlap = query_set.intersection(text_set)
        return float(len(overlap))

    def search_hybrid(
        self,
        query: str,
        project_id: str,
        limit: int = 5,
        entity_type: Optional[str] = None,
        k: int = 60,
        similarity_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Blends keyword search (exact overlap match) and vector lookup on LanceDB knowledge table.
        Applies a cosine similarity threshold to vector results (default 0.45 — inclusive for
        natural-language task queries). Uses Reciprocal Rank Fusion (RRF) to merge ranks.

        Args:
            query: Natural language search query.
            project_id: Scope all results to this project.
            limit: Max number of results to return after fusion.
            entity_type: If set, restrict results to this entity type (e.g. "TASK", "DOCUMENT").
            k: RRF constant — higher values flatten rank differences.
            similarity_threshold: Minimum cosine similarity for vector results to be included.
                                   Lower values (0.35–0.50) are better for general backlog queries.
        """
        logger.info(f"Hybrid search: query='{query}', project_id='{project_id}', entity_type='{entity_type}', threshold={similarity_threshold}")
        
        # 1. Vector Search
        vector = self.embedder.compute_embedding(query, "")
        vector_results = self.lancedb_manager.search_knowledge_similar(
            vector=vector,
            project_id=project_id,
            limit=limit * 3,
            entity_type=entity_type
        )
        
        ranked_vector_items = []
        for r in vector_results:
            distance = r.get('_distance', 2.0)
            similarity = 1.0 - (distance / 2.0)
            if similarity < similarity_threshold:
                logger.info(f"Skipping vector result (similarity {similarity:.4f} < threshold {similarity_threshold})")
                continue
            
            metadata_str = r.get("metadata", "{}")
            try:
                metadata = json.loads(metadata_str)
            except Exception:
                metadata = {}
                
            if "citation_hash" not in metadata:
                metadata["citation_hash"] = f"cit_{uuid.uuid4().hex[:5]}"
                
            ranked_vector_items.append({
                "entity_type": r.get("entity_type", "UNKNOWN"),
                "source_id": r.get("source_id", ""),
                "project_id": r.get("project_id", ""),
                "created_at": r.get("created_at", ""),
                "content_snippet": r.get("content_snippet", ""),
                "metadata": metadata,
                "similarity": similarity
            })

        # 2. Keyword Search — respect entity_type filter if provided
        filter_expr = f"project_id = '{project_id}'"
        if entity_type:
            filter_expr += f" AND entity_type = '{entity_type}'"
            
        all_records = []
        try:
            all_records = self.lancedb_manager.knowledge_table.search().where(filter_expr).to_list()
        except Exception as e:
            logger.error(f"Failed to fetch records for keyword search: {str(e)}")
            
        query_tokens = self._tokenize(query)
        scored_keyword_items = []
        
        for r in all_records:
            content_snippet = r.get("content_snippet", "")
            tokens = self._tokenize(content_snippet)
            score = self._score_keyword_match(query_tokens, tokens)
            if score > 0:
                metadata_str = r.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_str)
                except Exception:
                    metadata = {}
                    
                if "citation_hash" not in metadata:
                    metadata["citation_hash"] = f"cit_{uuid.uuid4().hex[:5]}"
                    
                scored_keyword_items.append((score, {
                    "entity_type": r.get("entity_type", "UNKNOWN"),
                    "source_id": r.get("source_id", ""),
                    "project_id": r.get("project_id", ""),
                    "created_at": r.get("created_at", ""),
                    "content_snippet": content_snippet,
                    "metadata": metadata,
                }))
                
        # Sort keyword items by score descending
        scored_keyword_items.sort(key=lambda x: x[0], reverse=True)
        ranked_keyword_items = [item for _, item in scored_keyword_items[:limit * 3]]

        # 3. Reciprocal Rank Fusion
        fused_items = self._reciprocal_rank_fusion(
            [ranked_vector_items, ranked_keyword_items],
            k=k
        )
        
        return fused_items[:limit]

    def _reciprocal_rank_fusion(self, rank_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        if not rank_lists:
            return []

        unique_keys = []
        key_to_item = {}
        
        for lst in rank_lists:
            for item in lst:
                if not item:
                    continue
                key = (item["entity_type"], item["source_id"])
                if key not in key_to_item:
                    unique_keys.append(key)
                    key_to_item[key] = item
                else:
                    existing = key_to_item[key]
                    if "similarity" not in existing and "similarity" in item:
                        existing["similarity"] = item["similarity"]

        num_items = len(unique_keys)
        if num_items == 0:
            return []

        rrf_scores = np.zeros(num_items)
        
        for lst in rank_lists:
            for rank_idx, item in enumerate(lst):
                if not item:
                    continue
                key = (item["entity_type"], item["source_id"])
                item_idx = unique_keys.index(key)
                rank = rank_idx + 1
                rrf_scores[item_idx] += 1.0 / (k + rank)

        sorted_indices = np.argsort(-rrf_scores)
        
        sorted_items = []
        for idx in sorted_indices:
            key = unique_keys[idx]
            item = key_to_item[key]
            item["rrf_score"] = float(rrf_scores[idx])
            sorted_items.append(item)
            
        return sorted_items

