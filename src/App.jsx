import { lazy, Suspense } from 'react'
import './App.css'
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom'
import AOS from 'aos';
import 'aos/dist/aos.css';

// Lazy load components
const Homepage = lazy(() => import('./Profile/Homepage.jsx'))
const PassionPage = lazy(() => import('./Passion/PassionPage.jsx'))
const Workex = lazy(() => import('./WorkExperience/Workex.jsx'))
const Sidebar = lazy(() => import('./Profile/Sidebar.jsx'))

// Initialize AOS with optimized settings
AOS.init({
  // Customize AOS to be less resource-intensive
  disable: 'mobile', // Disable on mobile to improve performance
  once: true,        // Only animate elements once
  mirror: false,     // No mirroring animations
  throttleDelay: 99, // Delay for throttling
});

function App() {
  return (
    <HashRouter>
      <Suspense fallback={<div className="loading">Loading...</div>}>
        <Sidebar />
        <div className="main-content">
          <Routes>
            {/* Main route for the portfolio homepage */}
            <Route path="/" element={<Homepage />} />
            
            {/* Route for the passions page */}
            <Route path="/passions" element={<PassionPage />} />

            {/* Route for the new workex page */}
            <Route path="/Workex" element={<Workex />} />
            
            {/* Redirect '/intro' to the homepage with a hash */}
            <Route path="/intro" element={<Navigate to="/#intro" replace />} />
            
            {/* Redirect any other unknown routes to the homepage */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </Suspense>
    </HashRouter>
  )
}

export default App;