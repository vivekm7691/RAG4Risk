/**
 * API client service for RAG4Risk frontend
 */

import axios from 'axios';

const API_BASE_URL = (import.meta.env?.VITE_API_BASE_URL as string) || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth tokens or other headers here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle common errors here
    if (error.response?.status === 401) {
      // Handle unauthorized
      console.error('Unauthorized access');
    }
    return Promise.reject(error);
  }
);

// API methods (to be implemented in Phase 3)
export const documentAPI = {
  // TODO: Implement in Phase 3
  upload: async (file: File, projectName: string, documentType: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('project_name', projectName);
    formData.append('document_type', documentType);
    
    return apiClient.post('/api/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  list: async (projectName?: string) => {
    return apiClient.get('/api/documents', {
      params: { project_name: projectName },
    });
  },
  
  get: async (documentId: string) => {
    return apiClient.get(`/api/documents/${documentId}`);
  },
  
  delete: async (documentId: string) => {
    return apiClient.delete(`/api/documents/${documentId}`);
  },
};

export const queryAPI = {
  // TODO: Implement in Phase 3
  query: async (query: string, projectName?: string, topK?: number) => {
    return apiClient.post('/api/query', {
      query,
      project_name: projectName,
      top_k: topK,
    });
  },
};

export default apiClient;

