import { NavLink, Outlet } from 'react-router-dom';
import './Layout.css';

export function Layout() {
  return (
    <div className="app-layout">
      <header className="app-header">
        <NavLink to="/" className="app-brand" end>
          RAG4Risk
        </NavLink>
        <nav className="app-nav" aria-label="Main">
          <NavLink to="/" className={({ isActive }) => (isActive ? 'active' : '')} end>
            Chat
          </NavLink>
          <NavLink to="/upload" className={({ isActive }) => (isActive ? 'active' : '')}>
            Upload
          </NavLink>
          <NavLink to="/documents" className={({ isActive }) => (isActive ? 'active' : '')}>
            Documents
          </NavLink>
          <NavLink to="/projects" className={({ isActive }) => (isActive ? 'active' : '')}>
            Projects
          </NavLink>
        </nav>
      </header>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
