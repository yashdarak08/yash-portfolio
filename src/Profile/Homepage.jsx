import React, { useEffect, useState } from 'react'
import ProfileIntro from './ProfileIntro';
import ProfileSkills from './ProfileSkills';
import ProfileProjects from './ProfileProjects';
import "./HomepageCSS.css";

export default function Homepage() {
    document.title = "Yash Darak";
    const [isScrolled, setIsScrolled] = useState(false);
    const isMobile = window.matchMedia("only screen and (max-width: 800px)").matches;

    useEffect(() => {
        // Handle scroll event to track scroll position
        const handleScroll = () => {
            // Check if we're scrolled down more than 100px
            const scrolled = window.scrollY > 100;
            setIsScrolled(scrolled);
            
            // Add or remove a class to the body when at the very top
            if (window.scrollY === 0) {
                document.body.classList.add('at-top');
            } else {
                document.body.classList.remove('at-top');
            }
        };

        // Set initial state
        document.body.classList.add('at-top');
        
        // This is for the viewport
        // If its a mobile device, then set the viewport to 800px
        setTimeout(() => {
            if(window.location.hash){
                const anchor = document.querySelector(window.location.hash)
                if(anchor){
                    console.log(anchor)
                    anchor.scrollIntoView({ behavior: 'smooth', block: 'center' })
                }
            }
        }, 200);

        // Add scroll event listener
        window.addEventListener('scroll', handleScroll);

        const currentMeta = document.getElementsByTagName('meta')['viewport'].content;

        if(isMobile){ 
            document.getElementsByTagName('meta')['viewport'].content='width=800, user-scalable=yes';
        }
        else{
            document.getElementsByTagName('meta')['viewport'].content='width=device-width, initial-scale=1';
        }

        return () => {
            // Clean up event listener
            window.removeEventListener('scroll', handleScroll);
            document.getElementsByTagName('meta')['viewport'].content = currentMeta;
        }
    }, [isMobile]);

    return (
        <div style={{padding: "10px 0px 0px 0px", overflow:"hidden", width: '100%'}}>
            <div id="intro" className={isScrolled ? "scrolled-header" : ""}>
                <ProfileIntro isMobile={isMobile} isScrolled={isScrolled} />
            </div>
            <div id="skills">
                <ProfileSkills />
            </div>
            <div id="projects">
                <ProfileProjects isMobile={false} />
            </div>
        </div>
    )
}