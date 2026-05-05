import React, { useState } from 'react';
import API from '../api';

export default function Login({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [err, setErr] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async e => {
    e.preventDefault();
    setErr('');
    setLoading(true);
    try {
      const res = await API.post('/auth/login', { email, password });
      const { token, user } = res.data;
      localStorage.setItem('wbc_user', JSON.stringify({ token, user }));
      onLogin(user, token);
    } catch (err) {
      setErr(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      maxWidth: '450px',
      margin: '60px auto',
      animation: 'fadeIn 0.5s ease'
    }}>
      <div style={{
        textAlign: 'center',
        marginBottom: '30px'
      }}>
        <div style={{
          fontSize: '4em',
          marginBottom: '10px'
        }}>🔬</div>
        <h2 className="text-gradient" style={{
          fontSize: '2.2em',
          marginBottom: '10px'
        }}>Welcome Back</h2>
        <p style={{ color: 'var(--color-text-muted)' }}>
          Sign in to access the diagnostic system
        </p>
      </div>

      <div className="card" style={{
        padding: '40px',
        boxShadow: 'var(--shadow-2xl)'
      }}>
        <form onSubmit={submit}>
          <label>📧 Email Address</label>
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            required
          />

          <label>🔒 Password</label>
          <input
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
          />

          <button
            type="submit"
            style={{ width: '100%', marginTop: '10px' }}
            disabled={loading}
          >
            {loading ? <span className="loading"></span> : '🚀 Login'}
          </button>

          {err && <div className="error">{err}</div>}
        </form>
      </div>
    </div>
  );
}
