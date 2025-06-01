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
const isMobile = window.matchMedia("(max-width: 800px)").matches;

AOS.init({
  disable: false,
  once: true,
  mirror: false,
  
  // Different settings for mobile vs desktop
  throttleDelay: isMobile ? 300 : 100,
  debounceDelay: isMobile ? 300 : 50,
  duration: isMobile ? 200 : 600,        // Much faster on mobile
  easing: isMobile ? 'ease' : 'ease-out',
  offset: isMobile ? 20 : 50,             // Trigger earlier on mobile
  
  // Mobile-specific optimizations
  startEvent: 'DOMContentLoaded',
  animatedClassName: 'aos-animate',
  initClassName: 'aos-init',
});

// Loading component with better mobile support
const LoadingComponent = () => (
  <div className="loading" style={{
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '100vh',
    background: 'var(--primaryBackground)',
    color: 'var(--primaryColor)',
    fontSize: '18px'
  }}>
    Loading...
  </div>
);

function App() {
  return (
    <HashRouter>
      <Suspense fallback={<LoadingComponent />}>
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