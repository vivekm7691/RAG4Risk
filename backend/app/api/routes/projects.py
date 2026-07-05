"""Project metadata API routes (Phase 3.5)."""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from app.models.document import ProjectMetadata, ProjectMetadataCreateUpdate
from app.services.project_metadata import ProjectMetadataService
from app.services.project_similarity import get_shared_project_similarity_service
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
metadata_service = ProjectMetadataService()
similarity_service = get_shared_project_similarity_service()


# Register the collection list before /{project_name}/... so paths like GET /api/projects
# are never captured by a dynamic segment (defensive ordering).
@router.get("", response_model=List[ProjectMetadata])
async def list_projects():
    """List all projects with metadata."""
    return metadata_service.get_all_projects()


@router.post("/metadata", response_model=ProjectMetadata, status_code=status.HTTP_200_OK)
async def create_or_update_project_metadata(data: ProjectMetadataCreateUpdate):
    """Create or update project metadata (upsert by project_name)."""
    try:
        out = metadata_service.create_or_update_project_metadata(data)
        similarity_service.clear_similarity_cache()
        if settings.GRAPH_ENABLED:
            try:
                from app.services.graph_sync_service import sync_project_metadata_to_graph

                await sync_project_metadata_to_graph(out)
            except Exception as graph_exc:
                logger.warning("Graph project metadata sync failed: %s", graph_exc)
        return out
    except Exception as e:
        logger.exception("Failed to create/update project metadata")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/{project_name}/metadata", response_model=ProjectMetadata)
async def get_project_metadata(project_name: str):
    """Get project metadata by project name."""
    meta = metadata_service.get_project_metadata(project_name)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return meta


@router.get("/{project_name}/similar")
async def get_similar_projects(project_name: str, top_k: int = 3):
    """Get projects similar to the given project (hybrid semantic + metadata)."""
    try:
        from app.services.vector_store import get_vector_store
        vector_store = await get_vector_store()
        results = await similarity_service.find_similar_projects(
            project_name=project_name,
            top_k=top_k,
            vector_store=vector_store,
        )
        return {"project_name": project_name, "similar_projects": results}
    except Exception as e:
        logger.exception("Failed to get similar projects")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
