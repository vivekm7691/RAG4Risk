"""Document upload and management API routes"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse

from app.models.document import (
    DocumentType,
    DocumentUpload,
    DocumentMetadata,
    DocumentResponse
)
from app.services.document_parser import DocumentParser
from app.services.excel_parser import ExcelParser
from app.services.chunker import Chunker
from app.services.embeddings import EmbeddingService
from app.services.vector_store import get_vector_store
from app.config import settings

router = APIRouter()

# Initialize services
document_parser = DocumentParser()
excel_parser = ExcelParser()
chunker = Chunker()
embedding_service = EmbeddingService()
# Note: vector_store is now async, get it per request

# Temporary file storage directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    project_name: str = Form(...),
    document_type: str = Form(...)
):
    """
    Upload a document (Word or Excel) and process it into the vector database
    
    - **file**: Document file (.docx for Word, .xlsx for Excel)
    - **project_name**: Name of the project this document belongs to
    - **document_type**: Type of document (statement of work, solution description document, 
      proposal document, risk register, issue log)
    """
    try:
        # Validate document type
        try:
            doc_type = DocumentType(document_type.lower())
        except ValueError:
            valid_types = [dt.value for dt in DocumentType]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid document_type. Must be one of: {', '.join(valid_types)}"
            )
        
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed extensions: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Validate file extension matches document type
        if doc_type in [DocumentType.RISK_REGISTER, DocumentType.ISSUE_LOG]:
            if file_ext != ".xlsx":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Excel document types require .xlsx files, got {file_ext}"
                )
        else:
            if file_ext != ".docx":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Word document types require .docx files, got {file_ext}"
                )
        
        # Validate file size
        file_content = await file.read()
        file_size = len(file_content)
        if file_size > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE} bytes"
            )
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Save file temporarily
        temp_file_path = UPLOAD_DIR / f"{document_id}{file_ext}"
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        try:
            # Parse document based on type
            if doc_type in [DocumentType.RISK_REGISTER, DocumentType.ISSUE_LOG]:
                # Excel file processing
                parse_result = excel_parser.parse(str(temp_file_path), doc_type)
                rows = parse_result["rows"]
                file_metadata = parse_result["metadata"]
                
                # Process Excel rows - each row is a chunk
                chunks = chunker.chunk_excel_rows(rows, {
                    "document_id": document_id,
                    "project_name": project_name,
                    "document_type": doc_type.value,
                    "file_name": file.filename,
                    "file_size": file_size,
                    "upload_date": datetime.utcnow().isoformat(),
                    **file_metadata
                })
                
                row_count = len(chunks)
            else:
                # Word document processing
                parse_result = document_parser.parse(str(temp_file_path), doc_type)
                text = parse_result["text"]
                file_metadata = parse_result["metadata"]
                
                # Chunk the document
                chunks = chunker.chunk_document(text, {
                    "document_id": document_id,
                    "project_name": project_name,
                    "document_type": doc_type.value,
                    "file_name": file.filename,
                    "file_size": file_size,
                    "upload_date": datetime.utcnow().isoformat(),
                    **file_metadata
                }, doc_type)
                
                row_count = None  # Not applicable for Word documents
            
            if not chunks:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No content extracted from document"
                )
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = embedding_service.generate_embeddings(chunk_texts)
            embeddings_list = embeddings.tolist()
            
            # Store in vector database (now async with Qdrant)
            vector_store = await get_vector_store()
            added_ids = await vector_store.add_documents(chunks, embeddings_list)
            
            # Prepare response metadata
            response_metadata = DocumentMetadata(
                document_id=document_id,
                project_name=project_name,
                document_type=doc_type,
                title=file_metadata.get("title") or file.filename,
                author=file_metadata.get("author"),
                upload_date=datetime.utcnow(),
                file_name=file.filename,
                file_size=file_size
            )
            
            # Prepare response message
            if row_count is not None:
                message = f"Document processed successfully. {row_count} rows processed, {len(added_ids)} new chunks added to vector database."
            else:
                message = f"Document processed successfully. {len(chunks)} chunks created, {len(added_ids)} new chunks added to vector database."
            
            return DocumentResponse(
                document_id=document_id,
                status="success",
                message=message,
                metadata=response_metadata
            )
            
        finally:
            # Clean up temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@router.get("", response_model=List[DocumentMetadata])
async def list_documents(project_name: Optional[str] = None):
    """
    List all documents, optionally filtered by project name
    
    - **project_name**: Optional project name filter
    """
    try:
        # Build where clause for filtering
        where_clause = None
        if project_name:
            where_clause = {"project_name": project_name}
        
        # Get all chunks from vector store (with optional project filter)
        # We'll extract unique documents from the chunks
        # Note: Qdrant uses async scroll with filters
        try:
            vector_store = await get_vector_store()
            
            # Build filter for project name if provided
            qdrant_filter = None
            if where_clause and "project_name" in where_clause:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="project_name",
                            match=MatchValue(value=where_clause["project_name"])
                        )
                    ]
                )
            
            # Scroll to get all points
            points, _ = await vector_store.client.scroll(
                collection_name=vector_store.collection_name,
                scroll_filter=qdrant_filter,
                limit=10000  # Large limit to get all points
            )
            
            # Convert points to results format
            results = {
                "ids": [point.id for point in points],
                "documents": [point.payload.get("text", "") for point in points],
                "metadatas": [{k: v for k, v in point.payload.items() if k != "text"} for point in points]
            }
        except Exception as e:
            # If collection is empty or query fails, return empty list
            return []
        
        # Extract unique documents from chunks
        documents_map = {}
        
        if results["ids"] and len(results["ids"]) > 0:
            for idx in range(len(results["ids"])):
                metadata = results["metadatas"][idx]
                doc_id = metadata.get("document_id")
                
                # Skip if we've already seen this document
                if doc_id in documents_map:
                    continue
                
                # Create document metadata
                try:
                    doc_metadata = DocumentMetadata(
                        document_id=doc_id,
                        project_name=metadata.get("project_name", ""),
                        document_type=DocumentType(metadata.get("document_type", "")),
                        title=metadata.get("title") or metadata.get("file_name", "Unknown"),
                        author=metadata.get("author"),
                        upload_date=datetime.fromisoformat(
                            metadata.get("upload_date", datetime.utcnow().isoformat())
                        ),
                        file_name=metadata.get("file_name", ""),
                        file_size=int(metadata.get("file_size", 0))
                    )
                    documents_map[doc_id] = doc_metadata
                except (ValueError, KeyError) as e:
                    # Skip documents with invalid metadata
                    continue
        
        return list(documents_map.values())
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.get("/{doc_id}", response_model=DocumentMetadata)
async def get_document(doc_id: str):
    """
    Get document metadata by document ID
    
    - **doc_id**: Document ID
    """
    try:
        # Get document chunks to extract metadata (now async with Qdrant)
        vector_store = await get_vector_store()
        chunks = await vector_store.get_document_chunks(doc_id)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {doc_id} not found"
            )
        
        # Extract metadata from first chunk (all chunks have same document metadata)
        metadata = chunks[0]["metadata"]
        
        return DocumentMetadata(
            document_id=metadata.get("document_id", doc_id),
            project_name=metadata.get("project_name", ""),
            document_type=DocumentType(metadata.get("document_type", "")),
            title=metadata.get("title"),
            author=metadata.get("author"),
            upload_date=datetime.fromisoformat(metadata.get("upload_date", datetime.utcnow().isoformat())),
            file_name=metadata.get("file_name", ""),
            file_size=int(metadata.get("file_size", 0))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str):
    """
    Delete a document and all its chunks from the vector database
    
    - **doc_id**: Document ID to delete
    """
    try:
        # Delete document (now async with Qdrant)
        vector_store = await get_vector_store()
        success = await vector_store.delete_document(doc_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {doc_id} not found"
            )
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

