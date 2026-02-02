"""Word document parsing service for RAG4Risk"""

from typing import Optional, Dict, Any
from datetime import datetime
from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

from app.models.document import DocumentType as DocType


class DocumentParser:
    """Parser for Word documents (.docx)"""
    
    def __init__(self):
        """Initialize the document parser"""
        pass
    
    def parse(self, file_path: str, document_type: DocType) -> Dict[str, Any]:
        """
        Parse a Word document and extract text content and metadata
        
        Args:
            file_path: Path to the .docx file
            document_type: Type of document being parsed
            
        Returns:
            Dictionary containing:
                - text: Full text content of the document
                - metadata: Document metadata (title, author, creation_date, etc.)
                
        Raises:
            ValueError: If file is not a valid .docx file
            FileNotFoundError: If file does not exist
        """
        try:
            doc = Document(file_path)
        except Exception as e:
            raise ValueError(f"Failed to parse Word document: {str(e)}")
        
        # Extract text content
        text_content = self._extract_text(doc)
        
        # Extract metadata
        metadata = self._extract_metadata(doc, document_type)
        
        return {
            "text": text_content,
            "metadata": metadata
        }
    
    def _extract_text(self, doc: DocumentType) -> str:
        """
        Extract text content from document, preserving structure
        
        Args:
            doc: python-docx Document object
            
        Returns:
            Full text content as a string
        """
        text_parts = []
        
        # Extract text from paragraphs and tables
        for element in doc.element.body:
            if isinstance(element, CT_P):
                # Paragraph
                para = Paragraph(element, doc)
                text = para.text.strip()
                if text:
                    text_parts.append(text)
            elif isinstance(element, CT_Tbl):
                # Table
                table = Table(element, doc)
                table_text = self._extract_table_text(table)
                if table_text:
                    text_parts.append(table_text)
        
        return "\n\n".join(text_parts)
    
    def _extract_table_text(self, table: Table) -> str:
        """
        Extract text from a table
        
        Args:
            table: python-docx Table object
            
        Returns:
            Table text as formatted string
        """
        table_rows = []
        for row in table.rows:
            row_cells = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    row_cells.append(cell_text)
            if row_cells:
                table_rows.append(" | ".join(row_cells))
        
        return "\n".join(table_rows) if table_rows else ""
    
    def _extract_metadata(self, doc: DocumentType, document_type: DocType) -> Dict[str, Any]:
        """
        Extract metadata from document
        
        Args:
            doc: python-docx Document object
            document_type: Type of document
            
        Returns:
            Dictionary of metadata
        """
        # Extract core properties
        core_props = doc.core_properties
        
        # Get title (from core properties or first heading)
        title = core_props.title
        if not title:
            # Try to get from first paragraph if it looks like a title
            if doc.paragraphs:
                first_para = doc.paragraphs[0].text.strip()
                if first_para and len(first_para) < 200:
                    title = first_para
        
        # Get author
        author = core_props.author
        
        # Get creation date
        creation_date = core_props.created
        if creation_date:
            creation_date = creation_date.isoformat()
        
        return {
            "title": title or "Untitled Document",
            "author": author,
            "creation_date": creation_date,
            "document_type": document_type.value,
            "paragraph_count": len([p for p in doc.paragraphs if p.text.strip()]),
            "table_count": len(doc.tables)
        }


