import lancedb
import pyarrow as pa
import os
import json
from typing import List, Dict, Any, Optional

class LanceDBManager:
    def __init__(self, db_path: str = "./.lancedb_store"):
        self.db_path = db_path
        self.uri = os.path.abspath(db_path)
        self.db = lancedb.connect(self.uri)
        self.table_name = "semantic_tasks_registry"
        self.knowledge_table_name = "workspace_knowledge_index"
        
        # Legacy tasks registry schema
        self.schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), 768)), # ModernBERT base dimensionality layout
            pa.field("task_id", pa.string()),
            pa.field("project_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("status", pa.string())
        ])
        
        # Unified knowledge base registry schema
        self.knowledge_schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), 768)),
            pa.field("entity_type", pa.string()),  # "TASK", "DOCUMENT", "COMMENT"
            pa.field("source_id", pa.string()),    # task_id, doc_id, comment_id
            pa.field("project_id", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("content_snippet", pa.string()),
            pa.field("metadata", pa.string())      # JSON string
        ])
        
        self._get_or_create_tables()

    def _get_or_create_tables(self):
        # Legacy table
        try:
            self.table = self.db.open_table(self.table_name)
        except ValueError:
            self.table = self.db.create_table(self.table_name, schema=self.schema)
            
        # Unified knowledge table
        try:
            self.knowledge_table = self.db.open_table(self.knowledge_table_name)
        except ValueError:
            self.knowledge_table = self.db.create_table(self.knowledge_table_name, schema=self.knowledge_schema)

    # ================= Legacy Task Registry Methods =================
    def search_similar(self, vector: List[float], project_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        result = (
            self.table.search(vector)
            .where(f"project_id = '{project_id}'")
            .limit(limit)
            .to_list()
        )
        return result

    def insert_task(self, vector: List[float], task_id: str, project_id: str, title: str, status: str) -> None:
        if len(vector) != 768:
            raise ValueError(f"Vector dimensions must be exactly 768, got {len(vector)}")
        if not task_id:
            raise ValueError("task_id must be a non-empty string")
        if not project_id:
            raise ValueError("project_id must be a non-empty string")
        
        record = {
            "vector": vector,
            "task_id": task_id,
            "project_id": project_id,
            "title": title,
            "status": status
        }
        pyarrow_table = pa.Table.from_pylist([record], schema=self.schema)
        self.table.add(pyarrow_table)

    def upsert_task(self, vector: List[float], task_id: str, project_id: str, title: str, status: str) -> None:
        if len(vector) != 768:
            raise ValueError(f"Vector dimensions must be exactly 768, got {len(vector)}")
        if not task_id:
            raise ValueError("task_id must be a non-empty string")
        if not project_id:
            raise ValueError("project_id must be a non-empty string")
        
        record = {
            "vector": vector,
            "task_id": task_id,
            "project_id": project_id,
            "title": title,
            "status": status
        }
        pyarrow_table = pa.Table.from_pylist([record], schema=self.schema)
        
        self.table.merge_insert("task_id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(pyarrow_table)

    def delete_task(self, task_id: str) -> None:
        if not task_id:
            raise ValueError("task_id must be a non-empty string")
        escaped_task_id = task_id.replace("'", "''")
        self.table.delete(f"task_id = '{escaped_task_id}'")

    # ================= Unified Knowledge Index Methods =================
    def search_knowledge_similar(self, vector: List[float], project_id: str, limit: int = 5, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search the unified knowledge base with metadata filters.
        """
        filter_expr = f"project_id = '{project_id}'"
        if entity_type:
            filter_expr += f" AND entity_type = '{entity_type}'"
            
        result = (
            self.knowledge_table.search(vector)
            .where(filter_expr)
            .limit(limit)
            .to_list()
        )
        return result

    def insert_knowledge(self, vector: List[float], entity_type: str, source_id: str, project_id: str, created_at: str, content_snippet: str, metadata_struct: Dict[str, Any]) -> None:
        if len(vector) != 768:
            raise ValueError(f"Vector dimensions must be exactly 768, got {len(vector)}")
        if not source_id:
            raise ValueError("source_id must be a non-empty string")
        if not project_id:
            raise ValueError("project_id must be a non-empty string")
        
        metadata_json = json.dumps(metadata_struct)
        
        record = {
            "vector": vector,
            "entity_type": entity_type,
            "source_id": source_id,
            "project_id": project_id,
            "created_at": created_at,
            "content_snippet": content_snippet,
            "metadata": metadata_json
        }
        pyarrow_table = pa.Table.from_pylist([record], schema=self.knowledge_schema)
        self.knowledge_table.add(pyarrow_table)

    def upsert_knowledge_batch(self, project_id: str, entity_type: str, source_id: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Replaces all existing knowledge index entries for a specific source_id with the new list of chunks.
        """
        if not source_id:
            raise ValueError("source_id must be a non-empty string")
            
        # 1. Delete existing chunks for this source_id
        escaped_source_id = source_id.replace("'", "''")
        self.knowledge_table.delete(f"source_id = '{escaped_source_id}'")
        
        # 2. If new chunks are provided, insert them
        if not chunks:
            return
            
        records = []
        for chunk in chunks:
            vector = chunk["vector"]
            if len(vector) != 768:
                raise ValueError(f"Vector dimensions must be exactly 768, got {len(vector)}")
                
            records.append({
                "vector": vector,
                "entity_type": entity_type,
                "source_id": source_id,
                "project_id": project_id,
                "created_at": chunk.get("created_at", ""),
                "content_snippet": chunk.get("content_snippet", ""),
                "metadata": json.dumps(chunk.get("metadata_struct", {}))
            })
            
        pyarrow_table = pa.Table.from_pylist(records, schema=self.knowledge_schema)
        self.knowledge_table.add(pyarrow_table)

    def delete_knowledge(self, source_id: str) -> None:
        if not source_id:
            raise ValueError("source_id must be a non-empty string")
        escaped_source_id = source_id.replace("'", "''")
        self.knowledge_table.delete(f"source_id = '{escaped_source_id}'")

