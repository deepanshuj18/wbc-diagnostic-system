import React from 'react';
export default function Navbar({ user, onLogout }){
  return (
    <div className="nav">
      <div className="nav-left">WBC Diagnostic</div>
      <div className="nav-right">
        {user ? <><span>{user.name}</span><button onClick={onLogout}>Logout</button></> : null}
      </div>
    </div>
  );
}
