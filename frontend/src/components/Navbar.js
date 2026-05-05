import React, { useState, useEffect } from 'react';

export default function Navbar({ user, onLogout }) {
  const [theme, setTheme] = useState('light');

  // Load theme from localStorage on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
    document.documentElement.setAttribute('data-theme', savedTheme);
  }, []);

  // Toggle theme
  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
  };

  return (
    <div className="nav">
      <div className="nav-left">
        WBC Diagnostic
      </div>
      <div className="nav-right">
        {user && <span>👤 {user.name}</span>}
        <button 
          onClick={toggleTheme} 
          style={{
            background: 'transparent',
            border: '2px solid var(--glass-border)',
            padding: '8px 16px',
            fontSize: '1.2em',
            minWidth: 'auto',
            boxShadow: 'var(--shadow-sm)'
          }}
          title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
        >
          {theme === 'light' ? '🌙' : '☀️'}
        </button>
        {user && (
          <button onClick={onLogout} style={{minWidth: 'auto'}}>
            Logout
          </button>
        )}
      </div>
    </div>
  );
}
