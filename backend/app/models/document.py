"""Document models for RAG4Risk"""

from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """Document type enumeration"""
    STATEMENT_OF_WORK = "statement of work"
    SOLUTION_DESCRIPTION = "solution description document"
    PROPOSAL = "proposal document"


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

