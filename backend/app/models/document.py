"""Document models for RAG4Risk"""

from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """Document type enumeration"""
    STATEMENT_OF_WORK = "statement of work"
    SOLUTION_DESCRIPTION = "solution description document"
    PROPOSAL = "proposal document"
    RISK_REGISTER = "risk register"
    ISSUE_LOG = "issue log"


class DocumentUpload(BaseModel):
    """Model for document upload request"""
    project_name: str = Field(..., min_length=1, max_length=100, description="Project name")
    document_type: DocumentType = Field(..., description="Type of document")
    
    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, v: str) -> str:
        """Validate project name"""
        if not v.strip():
            raise ValueError("Project name cannot be empty")
        return v.strip()


class DocumentMetadata(BaseModel):
    """Document metadata"""
    document_id: str
    project_name: str
    document_type: DocumentType
    title: Optional[str] = None
    author: Optional[str] = None
    upload_date: datetime
    file_name: str
    file_size: int


class DocumentResponse(BaseModel):
    """Response model for document operations"""
    document_id: str
    status: str
    message: str
    metadata: Optional[DocumentMetadata] = None


# --- Phase 3.5: Past Projects Enhancement ---

class ProjectMetadata(BaseModel):
    """Project metadata for similarity and past-projects context"""
    project_name: str = Field(..., min_length=1, max_length=200, description="Project name (unique identifier)")
    customer: str = Field(..., min_length=1, max_length=200, description="Customer's name (unique)")
    csg_products: List[str] = Field(default_factory=list, description="CSG products used in project")
    csg_role: Optional[str] = Field(None, max_length=100, description="e.g. Prime Contractor, Subcontractor, Consultant")
    integration_complexity: Optional[str] = Field(None, max_length=50, description="e.g. Low, Medium, High")
    client_type: Optional[str] = Field(None, max_length=50, description="e.g. Enterprise, SMB, Government")
    project_size: Optional[str] = Field(None, max_length=50, description="e.g. Small, Medium, Large")
    project_complexity: Optional[str] = Field(None, max_length=50, description="e.g. Simple, Moderate, Complex")
    date_range: Optional[Dict[str, str]] = Field(None, description="start_date, end_date (ISO strings)")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ProjectMetadataCreateUpdate(BaseModel):
    """Request model for creating or updating project metadata"""
    project_name: str = Field(..., min_length=1, max_length=200)
    customer: str = Field(..., min_length=1, max_length=200)
    csg_products: List[str] = Field(default_factory=list)
    csg_role: Optional[str] = None
    integration_complexity: Optional[str] = None
    client_type: Optional[str] = None
    project_size: Optional[str] = None
    project_complexity: Optional[str] = None
    date_range: Optional[Dict[str, str]] = None

