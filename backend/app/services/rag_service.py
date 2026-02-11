"""RAG service for LLM integration with Ollama"""

import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
from app.config import settings


class RAGService:
    """Service for Retrieval-Augmented Generation using Ollama LLM"""
    
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the RAG service
        
        Args:
            base_url: Ollama base URL (defaults to settings.OLLAMA_BASE_URL)
            model: Ollama model name (defaults to settings.OLLAMA_MODEL)
        """
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT  # Configurable timeout from settings
    
    async def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        project_name: Optional[str] = None,
        stream: bool = False
    ):
        """
        Generate a response using RAG (Retrieval-Augmented Generation)
        
        Args:
            query: User query text
            context_chunks: List of relevant chunks from vector search
            project_name: Optional project name for context
            stream: If True, returns an async generator that yields response chunks. If False, returns complete response string.
            
        Returns:
            If stream=False: Generated response string from LLM, or fallback summary if Ollama unavailable
            If stream=True: AsyncGenerator that yields response text chunks
        """
        if not context_chunks:
            if stream:
                async def empty_generator():
                    yield "I couldn't find any relevant information to answer your question."
                return empty_generator()
            else:
                return "I couldn't find any relevant information to answer your question."
        
        # Format context from chunks
        context_text = self._format_context(context_chunks, project_name)
        
        # Build prompt
        prompt = self._build_prompt(query, context_text, project_name)
        
        # Try to call Ollama API, fallback to summary if unavailable
        try:
            if stream:
                # Return async streaming generator directly (don't await)
                return self._call_ollama_streaming(prompt)
            else:
                # Return complete response
                response = await self._call_ollama(prompt)
                return response
        except RuntimeError as e:
            # If Ollama is unavailable, provide a fallback response
            if "Failed to connect" in str(e) or "Network is unreachable" in str(e):
                fallback = self._generate_fallback_response(query, context_chunks, project_name)
                if stream:
                    async def fallback_generator():
                        yield fallback
                    return fallback_generator()
                else:
                    return fallback
            else:
                # Re-raise other errors
                raise
    
    def _format_context(
        self,
        chunks: List[Dict[str, Any]],
        project_name: Optional[str] = None
    ) -> str:
        """
        Format retrieved chunks as context text
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            project_name: Optional project name for context labeling
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for idx, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            chunk_text = chunk.get("text", "")
            
            # Build source citation
            doc_name = metadata.get("file_name", "Unknown document")
            doc_type = metadata.get("document_type", "")
            chunk_id = metadata.get("chunk_id", "")
            
            # For Excel documents, include row and sheet info
            source_info = f"[Source {idx}: {doc_name}"
            if doc_type in ["risk register", "issue log"]:
                row_num = metadata.get("row_number")
                sheet_name = metadata.get("sheet_name")
                if row_num:
                    source_info += f", Row {row_num}"
                if sheet_name:
                    source_info += f", Sheet: {sheet_name}"
            source_info += f"]"
            
            context_parts.append(f"{source_info}\n{chunk_text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        project_name: Optional[str] = None
    ) -> str:
        """
        Build the RAG prompt with context injection
        
        Args:
            query: User query
            context: Formatted context from retrieved chunks
            project_name: Optional project name
            
        Returns:
            Complete prompt string
        """
        project_context = ""
        if project_name:
            project_context = f" for the project '{project_name}'"
        
        prompt = f"""You are a helpful assistant that answers questions based on provided context documents{project_context}.

Context from documents:
{context}

Question: {query}

Instructions:
- Answer the question based ONLY on the information provided in the context above.
- If the context doesn't contain enough information to answer the question, say so clearly.
- Be concise and accurate.
- Cite specific sources when referencing information (use the [Source X: ...] markers from the context).
- If the question is about risks or issues, provide specific details from the context including severity, status, and other relevant metadata.

Answer:"""
        
        return prompt
    
    async def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API to generate response (non-streaming)
        
        Args:
            prompt: Complete prompt string
            
        Returns:
            Generated response text
            
        Raises:
            RuntimeError: If API call fails
        """
        try:
            # Ollama API endpoint
            url = f"{self.base_url}/api/generate"
            
            # Request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False  # Non-streaming for simplicity
            }
            
            # Use async client with context manager
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                # Read response
                result = response.json()
                
                # Extract response text
                if "response" in result:
                    return result["response"].strip()
                else:
                    raise RuntimeError("Unexpected response format from Ollama API")
                    
        except httpx.TimeoutException:
            raise RuntimeError(f"Request to Ollama timed out after {self.timeout} seconds")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API returned error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Ollama: {str(e)}")
    
    async def _call_ollama_streaming(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Call Ollama API to generate streaming response
        
        Args:
            prompt: Complete prompt string
            
        Yields:
            Response text chunks as they arrive
            
        Raises:
            RuntimeError: If API call fails
        """
        # Ollama API endpoint
        url = f"{self.base_url}/api/generate"
        
        # Request payload with streaming enabled
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True  # Enable streaming
        }
        
        # Use async client with context manager
        # The context manager will stay open until generator is exhausted
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                try:
                    response.raise_for_status()
                    
                    # Parse NDJSON (newline-delimited JSON) stream
                    # Context manager stays open until generator is exhausted
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        
                        try:
                            chunk_data = json.loads(line)
                            
                            # Extract response text from chunk
                            if "response" in chunk_data:
                                yield chunk_data["response"]
                            
                            # Check if done
                            if chunk_data.get("done", False):
                                break
                                
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue
                            
                except httpx.TimeoutException:
                    raise RuntimeError(f"Request to Ollama timed out after {self.timeout} seconds")
                except httpx.HTTPStatusError as e:
                    raise RuntimeError(f"Ollama API returned error: {e.response.status_code} - {e.response.text}")
                except httpx.RequestError as e:
                    raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
                except Exception as e:
                    raise RuntimeError(f"Unexpected error calling Ollama: {str(e)}")
    
    def _generate_fallback_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        project_name: Optional[str] = None
    ) -> str:
        """
        Generate a fallback response when Ollama is unavailable
        Returns a summary of the retrieved chunks
        
        Args:
            query: User query text
            context_chunks: List of relevant chunks
            project_name: Optional project name
            
        Returns:
            Fallback response summarizing the retrieved chunks
        """
        if not context_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        # Build a simple summary from the chunks
        summary_parts = [
            f"Based on the retrieved documents{' for project ' + project_name if project_name else ''}, here is relevant information:"
        ]
        
        for idx, chunk in enumerate(context_chunks[:3], 1):  # Limit to top 3 chunks
            metadata = chunk.get("metadata", {})
            chunk_text = chunk.get("text", "")
            
            doc_name = metadata.get("file_name", "Unknown document")
            doc_type = metadata.get("document_type", "")
            
            source_info = f"Source {idx}: {doc_name}"
            if doc_type in ["risk register", "issue log"]:
                row_num = metadata.get("row_number")
                if row_num:
                    source_info += f" (Row {row_num})"
            
            # Truncate chunk text if too long
            if len(chunk_text) > 300:
                chunk_text = chunk_text[:300] + "..."
            
            summary_parts.append(f"\n{source_info}:")
            summary_parts.append(chunk_text)
        
        summary_parts.append(
            f"\n\nNote: Ollama LLM is not available. This is a summary of the retrieved documents. "
            f"To get AI-generated responses, please ensure Ollama is running at {self.base_url}"
        )
        
        return "\n".join(summary_parts)

