"""Project metadata service for Phase 3.5 Past Projects Enhancement.

Stores and retrieves project metadata in SQLite (separate from document chunks in Qdrant).
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import List, Optional

from app.config import settings
from app.models.document import ProjectMetadata, ProjectMetadataCreateUpdate

logger = logging.getLogger(__name__)


class ProjectMetadataService:
    """Service for managing project metadata in SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.PROJECT_METADATA_DB_PATH
        self._init_db()

    def _init_db(self) -> None:
        """Ensure data directory exists and create table if needed."""
        dir_path = os.path.dirname(self.db_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_metadata (
                    project_name TEXT PRIMARY KEY,
                    customer TEXT NOT NULL,
                    csg_products TEXT,
                    csg_role TEXT,
                    integration_complexity TEXT,
                    client_type TEXT,
                    project_size TEXT,
                    project_complexity TEXT,
                    date_range TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def create_or_update_project_metadata(self, data: ProjectMetadataCreateUpdate) -> ProjectMetadata:
        """Create or update project metadata. Uses project_name as unique key."""
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT project_name, created_at FROM project_metadata WHERE project_name = ?",
                (data.project_name,)
            ).fetchone()
            created_at = now if not row else row[1]
            conn.execute("""
                INSERT INTO project_metadata (
                    project_name, customer, csg_products, csg_role,
                    integration_complexity, client_type, project_size, project_complexity,
                    date_range, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_name) DO UPDATE SET
                    customer = excluded.customer,
                    csg_products = excluded.csg_products,
                    csg_role = excluded.csg_role,
                    integration_complexity = excluded.integration_complexity,
                    client_type = excluded.client_type,
                    project_size = excluded.project_size,
                    project_complexity = excluded.project_complexity,
                    date_range = excluded.date_range,
                    updated_at = excluded.updated_at
            """, (
                data.project_name,
                data.customer,
                json.dumps(data.csg_products or []),
                data.csg_role,
                data.integration_complexity,
                data.client_type,
                data.project_size,
                data.project_complexity,
                json.dumps(data.date_range) if data.date_range else None,
                created_at,
                now,
            ))
            conn.commit()
        finally:
            conn.close()
        return self.get_project_metadata(data.project_name)

    def ensure_default_metadata(self, project_name: str, customer_fallback: str) -> Optional[ProjectMetadata]:
        """
        If no metadata row exists for project_name, create a minimal row so similarity / past-projects
        features have a baseline (customer from document title or project name).
        """
        if self.get_project_metadata(project_name):
            return None
        customer = (customer_fallback or "").strip() or project_name
        dto = ProjectMetadataCreateUpdate(project_name=project_name, customer=customer)
        return self.create_or_update_project_metadata(dto)

    def get_project_metadata(self, project_name: str) -> Optional[ProjectMetadata]:
        """Get project metadata by project name."""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT project_name, customer, csg_products, csg_role, integration_complexity, "
                "client_type, project_size, project_complexity, date_range, created_at, updated_at "
                "FROM project_metadata WHERE project_name = ?",
                (project_name,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_model(row)
        finally:
            conn.close()

    def get_all_projects(self) -> List[ProjectMetadata]:
        """Return all stored project metadata."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT project_name, customer, csg_products, csg_role, integration_complexity, "
                "client_type, project_size, project_complexity, date_range, created_at, updated_at "
                "FROM project_metadata ORDER BY project_name"
            )
            return [self._row_to_model(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def _row_to_model(self, row: tuple) -> ProjectMetadata:
        project_name, customer, csg_products_json, csg_role, integration_complexity, \
        client_type, project_size, project_complexity, date_range_json, created_at, updated_at = row
        csg_products = json.loads(csg_products_json) if csg_products_json else []
        date_range = json.loads(date_range_json) if date_range_json else None
        return ProjectMetadata(
            project_name=project_name,
            customer=customer,
            csg_products=csg_products,
            csg_role=csg_role,
            integration_complexity=integration_complexity,
            client_type=client_type,
            project_size=project_size,
            project_complexity=project_complexity,
            date_range=date_range,
            created_at=datetime.fromisoformat(created_at) if created_at else None,
            updated_at=datetime.fromisoformat(updated_at) if updated_at else None,
        )
