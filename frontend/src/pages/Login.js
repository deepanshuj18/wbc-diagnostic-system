import React, { useState } from 'react';
import API from '../api';

export default function Login({ onLogin }){
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [err, setErr] = useState('');
  
  const submit = async e => {
    e.preventDefault();
    setErr('');
    try {
      const res = await API.post('/auth/login', { email, password });
      const { token, user } = res.data;
      localStorage.setItem('wbc_user', JSON.stringify({ token, user }));
      onLogin(user, token);
    } catch (err) {
      setErr(err.response?.data?.error || err.message);
    }
  };
  
  return (
    <div className="card" style={{maxWidth: '400px', margin: '0 auto'}}>
      <form onSubmit={submit}>
        <label>Email
          <input 
            type="email"
            value={email} 
            onChange={e=>setEmail(e.target.value)}
            required
          />
        </label>
        <label>Password
          <input 
            type="password" 
            value={password} 
            onChange={e=>setPassword(e.target.value)}
            required
          />
        </label>
        <button type="submit" style={{width: '100%'}}>Login</button>
        {err && <div className="error">{err}</div>}
      </form>
    </div>
  );
}
