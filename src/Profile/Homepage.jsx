import React, { useEffect } from 'react'
import ProfileIntro from './ProfileIntro';
import ProfileSkills from './ProfileSkills';
import ProfileProjects from './ProfileProjects';
import "./HomepageCSS.css";

export default function Homepage() {
    document.title = "Yash Darak";

    const isMobile = window.matchMedia("only screen and (max-width: 800px)").matches;
        useEffect(() => {
            // This is for the viewport
            // If its a mobile device, then set the viewport to 800px
            setTimeout(() => {
                if(window.location.hash && window.location.hash !== '#/') {
                    // Extract the section ID from the hash (remove the leading '#' or '#/')
                    const hashValue = window.location.hash.replace('#/', '').replace('#', '');
                    
                    // Only proceed if we have a non-empty hash
                    if (hashValue) {
                        const anchor = document.getElementById(hashValue);
                        if(anchor){
                            console.log('Scrolling to element with ID:', hashValue);
                            anchor.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        }
                    }
                }
            }, 200);

        // const currentMeta = document.getElementsByTagName('meta')['viewport'].content;

        // if(isMobile){ 
        //     document.getElementsByTagName('meta')['viewport'].content='width=800, user-scalable=yes';
        // }
        // else{
        //     document.getElementsByTagName('meta')['viewport'].content='width=device-width, initial-scale=1';
        // }

        // return () => {
        //     document.getElementsByTagName('meta')['viewport'].content = currentMeta;
        // }
    }, [isMobile]);

    return (
        <div style={{padding: "10px 0px 0px 0px", overflow:"hidden", width: '100%'}}>
            <div id="intro">
                <ProfileIntro isMobile={isMobile} />
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
