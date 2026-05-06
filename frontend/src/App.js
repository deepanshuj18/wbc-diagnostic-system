import React, { useState, useEffect } from 'react';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import PatientList from './pages/PatientList';

import ModelInsights from './pages/ModelInsights';
import Navbar from './components/Navbar';
import { setAuthToken } from './api';

function App(){
  const [user, setUser] = useState(null);
  const [view, setView] = useState('dashboard');
  const [showRegister, setShowRegister] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  
  useEffect(()=> {
    const raw = localStorage.getItem('wbc_user');
    if (raw) {
      const u = JSON.parse(raw);
      setUser(u.user);
      setAuthToken(u.token);
    }
  }, []);
  
  // Force refresh when switching to Dashboard
  const handleViewChange = (newView) => {
    setView(newView);
    if (newView === 'dashboard') {
      setRefreshKey(prev => prev + 1);
    }
  };

  const handleLogin = (userData, token) => {
    setUser(userData);
    setAuthToken(token);
  };

  const handleLogout = () => {
    localStorage.removeItem('wbc_user');
    setUser(null);
    setAuthToken(null);
    setView('dashboard');
  };

  if (!user) {
    return (
      <div>
        <Navbar user={null} />
        <div className="container">
          {!showRegister ? (
            <div>
              <h2>Login</h2>
              <Login onLogin={handleLogin} />
              <p style={{textAlign: 'center', marginTop: '20px'}}>
                Don't have an account? <a href="#" onClick={(e) => {e.preventDefault(); setShowRegister(true);}}>Register</a>
              </p>
            </div>
          ) : (
            <div>
              <h2>Register</h2>
              <Register onRegistered={() => setShowRegister(false)} />
              <p style={{textAlign: 'center', marginTop: '20px'}}>
                Already have an account? <a href="#" onClick={(e) => {e.preventDefault(); setShowRegister(false);}}>Login</a>
              </p>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div>
      <Navbar user={user} onLogout={handleLogout} />
      <div className="container">
        <nav className="view-nav">
          <button 
            className={view === 'dashboard' ? 'active' : ''} 
            onClick={() => handleViewChange('dashboard')}
          >
            Dashboard
          </button>
          <button 
            className={view === 'patients' ? 'active' : ''} 
            onClick={() => handleViewChange('patients')}
          >
            Patients
          </button>

          <button 
            className={view === 'insights' ? 'active' : ''} 
            onClick={() => handleViewChange('insights')}
          >
            Model Insights
          </button>
        </nav>
        
        <div className="view-content">
          {view === 'dashboard' && <Dashboard key={`dashboard-${refreshKey}`} />}
          {view === 'patients' && <PatientList key={view} />}

          {view === 'insights' && <ModelInsights key={view} />}
        </div>
      </div>
    </div>
  );
}

export default App;
