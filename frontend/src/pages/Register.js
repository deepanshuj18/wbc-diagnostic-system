import React, { useState } from 'react';
import API from '../api';

export default function Register({ onRegistered }){
  const [form, setForm] = useState({name:'', email:'', password:'', role:'clinician'});
  const [msg, setMsg] = useState('');
  
  const submit = async e => {
    e.preventDefault();
    setMsg('');
    try {
      const res = await API.post('/auth/register', form);
      setMsg('Registration successful! Please login.');
      setTimeout(() => {
        if (onRegistered) onRegistered();
      }, 1500);
    } catch (err) {
      setMsg(err.response?.data?.error || err.message);
    }
  };
  
  return (
    <div className="card" style={{maxWidth: '400px', margin: '0 auto'}}>
      <form onSubmit={submit}>
        <label>Name
          <input 
            type="text"
            value={form.name} 
            onChange={e=>setForm({...form, name:e.target.value})} 
            required
          />
        </label>
        <label>Email
          <input 
            type="email"
            value={form.email} 
            onChange={e=>setForm({...form, email:e.target.value})} 
            required
          />
        </label>
        <label>Password
          <input 
            type="password"
            value={form.password} 
            onChange={e=>setForm({...form, password:e.target.value})} 
            required
          />
        </label>
        <label>Role
          <select 
            value={form.role} 
            onChange={e=>setForm({...form, role:e.target.value})}
          >
            <option value="clinician">Clinician</option>
            <option value="admin">Admin</option>
          </select>
        </label>
        <button type="submit" style={{width: '100%'}}>Register</button>
        {msg && <div className={msg.includes('successful') ? 'success' : 'error'}>{msg}</div>}
      </form>
    </div>
  );
}
