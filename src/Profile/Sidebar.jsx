import React, { useState, useEffect } from 'react';
import { FaHome, FaTools, FaProjectDiagram, FaHeart, FaBriefcase } from 'react-icons/fa';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import './SidebarCSS.css';

export default function Sidebar() {
    const [activeSection, setActiveSection] = useState('intro');
    const location = useLocation();
    const navigate = useNavigate();
    const isHomePage = location.pathname === '/' || location.pathname === '';
    
    // Check for hash in the URL when component mounts
    useEffect(() => {
        if (location.hash) {
            // Remove the # from the hash
            const sectionId = location.hash.substring(1);
            setActiveSection(sectionId);
            
            // Small delay to ensure the DOM is fully loaded
            setTimeout(() => {
                const section = document.getElementById(sectionId);
                if (section) {
                    section.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }, 300);
        }
    }, [location.hash]);
    
    // Function to handle section navigation
    const scrollToSection = (sectionId) => {
        if (isHomePage) {
            const section = document.getElementById(sectionId);
            if (section) {
                section.scrollIntoView({ behavior: 'smooth' });
                setActiveSection(sectionId);
                
                // Update the URL hash without navigating
                window.history.replaceState(null, null, `#${sectionId}`);
            }
        } else {
            // If we're not on the homepage, navigate to the homepage with the hash
            navigate(`/#${sectionId}`);
        }
    };
    
    // Track scroll position when on homepage
    useEffect(() => {
        if (!isHomePage) return;
        
        const handleScroll = () => {
            const scrollPosition = window.scrollY;
            
            // Get all section elements
            const intro = document.getElementById('intro');
            const skills = document.getElementById('skills');
            const projects = document.getElementById('projects');
            
            // Calculate positions
            const introPos = intro ? intro.offsetTop : 0;
            const skillsPos = skills ? skills.offsetTop : 0;
            const projectsPos = projects ? projects.offsetTop : 0;
            
            // Set active section based on scroll position
            if (scrollPosition < skillsPos - 100) {
                setActiveSection('intro');
            } else if (scrollPosition < projectsPos - 100) {
                setActiveSection('skills');
            } else {
                setActiveSection('projects');
            }
        };
        
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, [isHomePage]);
    
    return (
        <div className="sidebar">
            <div className="sidebar-content">
                <div className="sidebar-logo">
                    <span>YD</span>
                </div>
                
                <nav className="sidebar-nav">
                    <ul>
                        {isHomePage ? (
                            // Navigation for the Home Page
                            <>
                                <li 
                                    className={activeSection === 'intro' ? 'active' : ''} 
                                    onClick={() => scrollToSection('intro')}
                                >
                                    <FaHome className="sidebar-icon" />
                                    <span className="sidebar-text">Intro</span>
                                </li>
                                <li 
                                    className={activeSection === 'skills' ? 'active' : ''} 
                                    onClick={() => scrollToSection('skills')}
                                >
                                    <FaTools className="sidebar-icon" />
                                    <span className="sidebar-text">Skills</span>
                                </li>
                                <li 
                                    className={activeSection === 'projects' ? 'active' : ''} 
                                    onClick={() => scrollToSection('projects')}
                                >
                                    <FaProjectDiagram className="sidebar-icon" />
                                    <span className="sidebar-text">Projects</span>
                                </li>
                                <li>
                                    <Link to="/passions" className="sidebar-link">
                                        <FaHeart className="sidebar-icon" />
                                        <span className="sidebar-text">Passions</span>
                                    </Link>
                                </li>
                                <li>    
                                    <Link to="/workex" className="sidebar-link">
                                        <FaProjectDiagram className="sidebar-icon" />
                                        <span className="sidebar-text">Work Ex</span>
                                    </Link>
                                </li>
                            </>
                        ) : (
                            // Navigation for other pages
                            <>
                                <li>
                                    <Link to="/" className="sidebar-link">
                                        <FaHome className="sidebar-icon" />
                                        <span className="sidebar-text">Home</span>
                                    </Link>
                                </li>
                                <li>
                                    <Link to="/passions" className={location.pathname === '/passions' ? 'sidebar-link active' : 'sidebar-link'}>
                                        <FaHeart className="sidebar-icon" />
                                        <span className="sidebar-text">Passions</span>
                                    </Link>
                                </li>
                                <li>
                                    <Link to="/workex" className={location.pathname === '/workex' ? 'sidebar-link active' : 'sidebar-link'}>
                                        <FaBriefcase className="sidebar-icon" />
                                        <span className="sidebar-text">Work Ex</span>
                                    </Link>
                                </li>
                            </>
                        )}
                    </ul>
                </nav>
            </div>
        </div>
    );
}