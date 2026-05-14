import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export function checkHealth() {
  return api.get('/health');
}

export function predictEmotion(file) {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
}
