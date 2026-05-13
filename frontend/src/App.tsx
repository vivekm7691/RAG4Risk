import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Layout } from './components/Layout';
import ChatInterface from './components/ChatInterface';
import { DocumentUpload } from './components/DocumentUpload';
import { DocumentList } from './components/DocumentList';
import { ProjectSelector } from './components/ProjectSelector';
import './App.css';

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <AppProvider>
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<Layout />}>
                <Route index element={<ChatInterface />} />
                <Route path="upload" element={<DocumentUpload />} />
                <Route path="documents" element={<DocumentList />} />
                <Route path="projects" element={<ProjectSelector />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </ErrorBoundary>
        </AppProvider>
      </BrowserRouter>
    </div>
  );
}

export default App;
