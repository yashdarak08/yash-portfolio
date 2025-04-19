import './App.css'
import Homepage from './Profile/Homepage.jsx'
import PassionPage from './Passion/PassionPage.jsx'
import Sidebar from './Profile/Sidebar.jsx'
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom'
import AOS from 'aos';
import 'aos/dist/aos.css';
AOS.init();

function App(props) {
  console.log(props)
  
  return (
    <HashRouter>
      <Sidebar />
      <div className="main-content">
        <Routes>
          {/* Main route for the portfolio homepage */}
          <Route path="/" element={<Homepage />} />
          
          {/* Route for the passions page */}
          <Route path="/passions" element={<PassionPage />} />
          
          {/* Redirect '/intro' to the homepage with a hash */}
          <Route path="/intro" element={<Navigate to="/#intro" replace />} />
          
          {/* Redirect any other unknown routes to the homepage */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </HashRouter>
  )
}

export default App;