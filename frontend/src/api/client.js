import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const authApi = {
  login: (email, password) => {
    const params = new URLSearchParams();
    params.append('username', email);
    params.append('password', password);
    return api.post('/auth/login', params, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
  },
  register: (email, username, password) =>
    api.post('/auth/register', { email, username, password }),
  me: () => api.get('/auth/me'),
};

export const scenariosApi = {
  list: () => api.get('/scenarios/'),
  listUnlocked: () => api.get('/scenarios/unlocked'),
  get: (id) => api.get(`/scenarios/${id}`),
  generateAdaptive: () => api.post('/scenarios/generate-adaptive'),
  generateFromUrl: (url, difficulty = 3) =>
    api.post('/scenarios/generate-from-url', null, { params: { url, difficulty } }),
};

export const simulationsApi = {
  list: (params) => api.get('/simulations/', { params }),
  start: (scenarioId) => api.post('/simulations/start', { scenario_id: scenarioId }),
  getState: (id, time) => api.get(`/simulations/${id}/state`, { params: { time_elapsed: time } }),
  makeDecision: (id, decision) => api.post(`/simulations/${id}/decision`, decision),
  complete: (id) => api.post(`/simulations/${id}/complete`, {}),
  abandon: (id) => api.post(`/simulations/${id}/abandon`),
  challenge: (id, data) => api.post(`/simulations/${id}/challenge`, data),
  skipTime: (id, seconds) => api.post(`/simulations/${id}/skip-time`, { seconds }),
  getStreamToken: (id) => api.post(`/simulations/${id}/stream-token`),
  verifyCredibility: (claim, sourceType = 'news') =>
    api.post('/simulations/verify-credibility', null, { params: { claim, source_type: sourceType } }),
};

// SSE streaming helper (uses opaque stream tokens — no JWT in query strings)
export const streamApi = {
  simulationStream: async (id) => {
    const res = await simulationsApi.getStreamToken(id);
    return fetch(`/api/simulations/${id}/stream?token=${res.data.token}`);
  },
};

export const reflectionApi = {
  get: (simulationId) => api.get(`/reflection/${simulationId}`),
  getFull: (simulationId) => api.get(`/reflection/${simulationId}/full`),
  getCounterfactuals: (simulationId) => api.get(`/reflection/${simulationId}/counterfactuals`),
  getWhyDecisions: (simulationId) => api.get(`/reflection/${simulationId}/why`),
  getProComparison: (simulationId) => api.get(`/reflection/${simulationId}/pro-comparison`),
  getCoaching: (simulationId) => api.get(`/reflection/${simulationId}/coaching`),
  getBiasHeatmap: (simulationId) => api.get(`/reflection/${simulationId}/bias-heatmap`),
  getRationaleReview: (simulationId) => api.get(`/reflection/${simulationId}/rationale-review`),
  getCounterfactualIsolation: (simulationId, decisionIndex) =>
    api.get(`/reflection/${simulationId}/counterfactual-isolation`, { params: { decision_index: decisionIndex } }),
  getCalibration: (simulationId) => api.get(`/reflection/${simulationId}/calibration`),
  getOutcomeDistribution: (simulationId) => api.get(`/reflection/${simulationId}/outcome-distribution`),
};

export const learningApi = {
  getCards: () => api.get('/learning/cards'),
  submitFeedback: (cardId, helpful) =>
    api.post('/learning/cards/feedback', { card_id: cardId, marked_helpful: helpful }),
  getModules: () => api.get('/learning/modules'),
  getModule: (moduleId) => api.get(`/learning/modules/${moduleId}`),
  completeModule: (moduleId) => api.post(`/learning/modules/${moduleId}/complete`),
};

export const profileApi = {
  get: () => api.get('/profile/'),
  getSummary: () => api.get('/profile/summary'),
  getHistory: () => api.get('/profile/history'),
  getPlaybook: () => api.get('/profile/playbook'),
  trackPlaybook: (simulationId) =>
    api.post('/profile/playbook/track', null, { params: { simulation_id: simulationId } }),
};

export default api;
