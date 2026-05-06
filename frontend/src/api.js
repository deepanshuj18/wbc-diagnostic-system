import axios from 'axios';

const API = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:4000/api'
});

export function setAuthToken(token) {
  if (token) API.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  else delete API.defaults.headers.common['Authorization'];
}

// New API functions for v2
export async function getModelEvaluation() {
  const res = await API.get('/models/evaluation');
  return res.data;
}

export async function getEmbeddingsVisualization() {
  const res = await API.get('/models/embeddings');
  return res.data;
}

export default API;
