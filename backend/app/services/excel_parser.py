"""Excel file parsing service for risk registers and issue logs"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import re
import pandas as pd
from pathlib import Path

from app.models.document import DocumentType as DocType


class ExcelParser:
    """Parser for Excel files (.xlsx) - risk registers and issue logs"""
    
    def __init__(self):
        """Initialize the Excel parser"""
        pass
    
    def parse(self, file_path: str, document_type: DocType) -> Dict[str, Any]:
        """
        Parse an Excel file and extract rows with metadata
        
        Args:
            file_path: Path to the .xlsx file
            document_type: Type of document ('risk register' or 'issue log')
            
        Returns:
            Dictionary containing:
                - rows: List of row objects with data and metadata
                - metadata: File-level metadata (sheet names, column headers, etc.)
                
        Raises:
            ValueError: If file is not a valid .xlsx file or document_type is invalid
            FileNotFoundError: If file does not exist
        """
        if document_type not in [DocType.RISK_REGISTER, DocType.ISSUE_LOG]:
            raise ValueError(f"Excel parser only supports 'risk register' and 'issue log' document types, got: {document_type.value}")
        
        try:
            # Map document_type to target sheet name
            sheet_mapping = {
                DocType.RISK_REGISTER: "Risk Log (Detailed Info)",
                DocType.ISSUE_LOG: "Issue Log (Detailed Info)"
            }
            
            target_sheet_name = sheet_mapping[document_type]
            
            # Read all sheets from Excel file
            excel_file = pd.ExcelFile(file_path)
            
            # Validate target sheet exists
            if target_sheet_name not in excel_file.sheet_names:
                raise ValueError(
                    f"Required sheet '{target_sheet_name}' not found in Excel file. "
                    f"Available sheets: {excel_file.sheet_names}"
                )
            
            # Filter to only process target sheet
            target_sheets = [target_sheet_name]
            
            all_rows = []
            file_metadata = {
                "sheet_names": target_sheets,
                "total_sheets": len(target_sheets),
                "document_type": document_type.value
            }
            
            # Process only the target sheet
            for sheet_name in target_sheets:
                # Skip rows 1-10, use row 11 as headers, process from row 12 onwards
                df = pd.read_excel(
                    excel_file,
                    sheet_name=sheet_name,
                    skiprows=range(0, 10),  # Skip Excel rows 1-10 (0-indexed: 0-9)
                    header=0  # Use first row of remaining data (Excel row 11) as headers
                )
                # After skipping: DataFrame index 0 = Excel row 11 (headers), index 1+ = Excel row 12+ (data)
                
                # Extract column headers
                column_headers = df.columns.tolist()
                
                # Process each row
                for idx, row in df.iterrows():
                    # Skip completely empty rows
                    if row.isna().all():
                        continue
                    
                    # Convert row to dictionary
                    row_data = row.to_dict()
                    
                    # Stop processing if this is a guidelines row
                    # Pass column_headers to check first column for numbers
                    if self._is_guidelines_row(row_data, column_headers):
                        break  # Stop processing remaining rows (exclude guidelines row)
                    
                    # Create row object with metadata
                    row_obj = self._create_row_object(
                        row_data=row_data,
                        row_number=idx + 12,  # DataFrame index 0 = Excel row 11, index 1 = Excel row 12, etc.
                        sheet_name=sheet_name,
                        column_headers=column_headers,
                        document_type=document_type
                    )
                    
                    all_rows.append(row_obj)
            
            file_metadata["total_rows"] = len(all_rows)
            file_metadata["row_count_by_sheet"] = {
                sheet: sum(1 for r in all_rows if r["metadata"]["sheet_name"] == sheet)
                for sheet in target_sheets
            }
            
            return {
                "rows": all_rows,
                "metadata": file_metadata
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse Excel file: {str(e)}")
    
    def _is_guidelines_row(
        self, 
        row_data: Dict[str, Any], 
        column_headers: List[str]
    ) -> bool:
        """
        Check if a row is a guidelines row based on specific pattern:
        1. Row contains 'Guidelines:' text (case-insensitive)
        2. First column does NOT contain a number (risk ID or issue ID)
        
        Args:
            row_data: Dictionary of column values for the row
            column_headers: List of column header names (to identify first column)
            
        Returns:
            True if row matches guidelines pattern, False otherwise
        """
        # Check if any cell contains 'Guidelines:' (case-insensitive)
        has_guidelines_text = False
        for value in row_data.values():
            if pd.notna(value):
                value_str = str(value).strip()
                if value_str.lower() == "guidelines:":
                    has_guidelines_text = True
                    break
        
        # If no 'Guidelines:' text found, not a guidelines row
        if not has_guidelines_text:
            return False
        
        # Check first column for number (risk ID or issue ID)
        # First column is the first key in row_data (or first column header)
        if not column_headers:
            return False
        
        first_column_name = column_headers[0]
        first_column_value = row_data.get(first_column_name)
        
        # Check if first column contains a number
        if pd.notna(first_column_value):
            first_col_str = str(first_column_value).strip()
            # Check if first column contains any digits (representing risk/issue ID)
            if re.search(r'\d', first_col_str):
                # First column has a number, so this is NOT a guidelines row
                # (it's actual data that happens to mention "Guidelines:")
                return False
        
        # Row has 'Guidelines:' and first column has no number - it's a guidelines row
        return True
    
    def _create_row_object(
        self,
        row_data: Dict[str, Any],
        row_number: int,
        sheet_name: str,
        column_headers: List[str],
        document_type: DocType
    ) -> Dict[str, Any]:
        """
        Create a structured row object with data and metadata
        
        Args:
            row_data: Dictionary of column values for the row
            row_number: Excel row number (1-indexed, accounting for header)
            sheet_name: Name of the Excel sheet
            column_headers: List of column header names
            document_type: Type of document
            
        Returns:
            Dictionary containing:
                - text: Formatted text representation of the row
                - metadata: Row-level metadata
        """
        # Convert row data to text format: "Column1: value1, Column2: value2, ..."
        text_parts = []
        for col, value in row_data.items():
            if pd.notna(value) and str(value).strip():
                text_parts.append(f"{col}: {value}")
        
        row_text = ", ".join(text_parts)
        
        # Extract common metadata fields (case-insensitive matching)
        metadata = {
            "row_number": row_number,
            "sheet_name": sheet_name,
            "document_type": document_type.value
        }
        
        # Extract Excel-specific metadata fields
        # Common fields for risk registers and issue logs
        field_mapping = {
            "risk_id": ["risk_id", "risk id", "id", "risk identifier"],
            "issue_id": ["issue_id", "issue id", "id", "issue identifier"],
            "severity": ["severity", "risk severity", "issue severity", "priority"],
            "status": ["status", "state", "current status"],
            "category": ["category", "type", "risk category", "issue category"],
            "date": ["date", "created date", "identified date", "date identified"],
            "owner": ["owner", "assigned to", "assignee", "responsible"],
            "description": ["description", "risk description", "issue description"],
            "mitigation": ["mitigation", "mitigation plan", "action plan"],
            "project_name": ["project_name", "project name", "project"]
        }
        
        # Map fields based on document type
        if document_type == DocType.RISK_REGISTER:
            id_fields = field_mapping["risk_id"]
        else:  # ISSUE_LOG
            id_fields = field_mapping["issue_id"]
        
        # Extract metadata fields (case-insensitive)
        row_data_lower = {str(k).lower(): v for k, v in row_data.items()}
        
        for metadata_key, possible_names in field_mapping.items():
            # Skip risk_id/issue_id based on document type
            if metadata_key in ["risk_id", "issue_id"]:
                if document_type == DocType.RISK_REGISTER and metadata_key == "issue_id":
                    continue
                if document_type == DocType.ISSUE_LOG and metadata_key == "risk_id":
                    continue
            
            for name in possible_names:
                if name.lower() in row_data_lower:
                    value = row_data_lower[name.lower()]
                    if pd.notna(value) and str(value).strip():
                        # Convert dates to ISO format if possible
                        if metadata_key == "date" and isinstance(value, (pd.Timestamp, datetime)):
                            metadata[metadata_key] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                        else:
                            metadata[metadata_key] = str(value).strip()
                        break
        
        # Store all column values as metadata for flexible filtering
        metadata["all_columns"] = {
            str(k): str(v) if pd.notna(v) else "" 
            for k, v in row_data.items()
        }
        
        return {
            "text": row_text,
            "metadata": metadata
        }


