import lancedb
import pyarrow as pa
import os
from typing import List, Dict, Any

class LanceDBManager:
    def __init__(self, db_path: str = "./.lancedb_store"):
        self.db_path = db_path
        self.uri = os.path.abspath(db_path)
        self.db = lancedb.connect(self.uri)
        self.table_name = "semantic_tasks_registry"
        self.schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), 768)), # ModernBERT base dimensionality layout
            pa.field("task_id", pa.string()),
            pa.field("project_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("status", pa.string())
        ])
        self._get_or_create_table()

    def _get_or_create_table(self):
        try:
            self.table = self.db.open_table(self.table_name)
        except ValueError:
            self.table = self.db.create_table(self.table_name, schema=self.schema)

    def search_similar(self, vector: List[float], project_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        # Enforce boundary separation constraints via metadata filtering rules
        result = (
            self.table.search(vector)
            .where(f"project_id = '{project_id}'")
            .limit(limit)
            .to_list()
        )
        return result

    def insert_task(self, vector: List[float], task_id: str, project_id: str, title: str, status: str) -> None:
        """
        Insertion: Write records dynamically via pyarrow structural blocks upon operational intercept validations.
        """
        # Validate boundary conditions and presence constraints
        if len(vector) != 768:
            raise ValueError(f"Vector dimensions must be exactly 768, got {len(vector)}")
        if not task_id:
            raise ValueError("task_id must be a non-empty string")
        if not project_id:
            raise ValueError("project_id must be a non-empty string")
        
        # Build PyArrow Table (structural block) for strict schema enforcement
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
        """
        Updates: Execute matching upsert sequences target-bounded by the tracking identifier key task_id.
        """
        # Validate inputs
        if len(vector) != 768:
            raise ValueError(f"Vector dimensions must be exactly 768, got {len(vector)}")
        if not task_id:
            raise ValueError("task_id must be a non-empty string")
        if not project_id:
            raise ValueError("project_id must be a non-empty string")
        
        # Build PyArrow Table matching schema
        record = {
            "vector": vector,
            "task_id": task_id,
            "project_id": project_id,
            "title": title,
            "status": status
        }
        pyarrow_table = pa.Table.from_pylist([record], schema=self.schema)
        
        # Perform merge insert (upsert) keyed by task_id
        self.table.merge_insert("task_id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(pyarrow_table)

    def delete_task(self, task_id: str) -> None:
        """
        Deletions: Execute deletions using local filter keys: table.delete("task_id = 'XYZ'").
        """
        if not task_id:
            raise ValueError("task_id must be a non-empty string")
        
        # Escape single quotes in task_id to prevent injection issues in search filter
        escaped_task_id = task_id.replace("'", "''")
        self.table.delete(f"task_id = '{escaped_task_id}'")
