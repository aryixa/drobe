import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import { Sidebar } from './components/layout/Sidebar'
import { Auth } from './pages/Auth'
import { Dashboard } from './pages/Dashboard'
import { Wardrobe } from './pages/Wardrobe'
import { Stylist } from './pages/Stylist'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth()

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-900"></div>
      </div>
    )
  }

  if (!user) {
    return <Navigate to="/auth" replace />
  }

  return <>{children}</>
}

function AppLayout() {
  const { signOut } = useAuth()
  const [isUploadModalOpen, setIsUploadModalOpen] = React.useState(false)

  return (
    <div className="flex h-screen bg-primary-50">
      <Sidebar 
        onUploadClick={() => setIsUploadModalOpen(true)}
        onSignOut={signOut}
      />
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/wardrobe" element={<Wardrobe />} />
          <Route path="/stylist" element={<Stylist />} />
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </main>
    </div>
  )
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/auth" element={<Auth />} />
      <Route
        path="/*"
        element={
          <ProtectedRoute>
            <AppLayout />
          </ProtectedRoute>
        }
      />
    </Routes>
  )
}

function App() {
  return (
    <AuthProvider>
      <Router>
        <AppRoutes />
      </Router>
    </AuthProvider>
  )
}

export default App
