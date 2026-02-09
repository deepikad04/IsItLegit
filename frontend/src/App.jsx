import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';
import Layout from './components/common/Layout';

// Lazy-load heavy pages (Recharts, Gemini analysis UI)
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Simulation = lazy(() => import('./pages/Simulation'));
const Reflection = lazy(() => import('./pages/Reflection'));
const Learning = lazy(() => import('./pages/Learning'));
const Profile = lazy(() => import('./pages/Profile'));

function LazyFallback() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
    </div>
  );
}

function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return children;
}

function AppRoutes() {
  const { user } = useAuth();

  return (
    <Routes>
      <Route path="/" element={user ? <Navigate to="/dashboard" replace /> : <Home />} />
      <Route path="/login" element={user ? <Navigate to="/dashboard" replace /> : <Login />} />
      <Route path="/register" element={user ? <Navigate to="/dashboard" replace /> : <Register />} />

      <Route path="/dashboard" element={
        <ProtectedRoute>
          <Layout>
            <Dashboard />
          </Layout>
        </ProtectedRoute>
      } />

      <Route path="/simulation/:scenarioId" element={
        <ProtectedRoute>
          <Simulation />
        </ProtectedRoute>
      } />

      <Route path="/reflection/:simulationId" element={
        <ProtectedRoute>
          <Layout>
            <Reflection />
          </Layout>
        </ProtectedRoute>
      } />

      <Route path="/learning" element={
        <ProtectedRoute>
          <Layout>
            <Learning />
          </Layout>
        </ProtectedRoute>
      } />

      <Route path="/profile" element={
        <ProtectedRoute>
          <Layout>
            <Profile />
          </Layout>
        </ProtectedRoute>
      } />
    </Routes>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Suspense fallback={<LazyFallback />}>
          <AppRoutes />
        </Suspense>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
