import React, { useState } from 'react';
import API from '../api';

export default function Register({ onRegistered }) {
  const [form, setForm] = useState({ name: '', email: '', password: '', role: 'clinician' });
  const [msg, setMsg] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async e => {
    e.preventDefault();
    setMsg('');
    setLoading(true);
    try {
      const res = await API.post('/auth/register', form);
      setMsg('Registration successful! Please login.');
      setTimeout(() => {
        if (onRegistered) onRegistered();
      }, 1500);
    } catch (err) {
      setMsg(err.response?.data?.error || err.message);
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
        }}>🧬</div>
        <h2 className="text-gradient" style={{
          fontSize: '2.2em',
          marginBottom: '10px'
        }}>Create Account</h2>
        <p style={{ color: 'var(--color-text-muted)' }}>
          Join the WBC diagnostic platform
        </p>
      </div>

      <div className="card" style={{
        padding: '40px',
        boxShadow: 'var(--shadow-2xl)'
      }}>
        <form onSubmit={submit}>
          <label>👤 Full Name</label>
          <input
            type="text"
            placeholder="Enter your full name"
            value={form.name}
            onChange={e => setForm({ ...form, name: e.target.value })}
            required
          />

          <label>📧 Email Address</label>
          <input
            type="email"
            placeholder="Enter your email"
            value={form.email}
            onChange={e => setForm({ ...form, email: e.target.value })}
            required
          />

          <label>🔒 Password</label>
          <input
            type="password"
            placeholder="Create a strong password"
            value={form.password}
            onChange={e => setForm({ ...form, password: e.target.value })}
            required
          />

          <label>👔 Role</label>
          <select
            value={form.role}
            onChange={e => setForm({ ...form, role: e.target.value })}
          >
            <option value="clinician">Clinician</option>
            <option value="admin">Admin</option>
          </select>

          <button
            type="submit"
            style={{ width: '100%', marginTop: '10px' }}
            disabled={loading}
          >
            {loading ? <span className="loading"></span> : '✨ Create Account'}
          </button>

          {msg && <div className={msg.includes('successful') ? 'success' : 'error'}>{msg}</div>}
        </form>
      </div>
    </div>
  );
}
