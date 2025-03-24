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
            
            if(window.location.hash){
                const anchor = document.querySelector(window.location.hash)
                if(anchor){
                    console.log(anchor)
                    anchor.scrollIntoView({ behavior: 'smooth', block: 'center' })
                }
            }
        }, 200);
        // console.log(window.location.hash)

        const currentMeta = document.getElementsByTagName('meta')['viewport'].content;

        const isMobile = window.matchMedia("only screen and (max-width: 800px)").matches;
        if(isMobile){ 
            document.getElementsByTagName('meta')['viewport'].content='width=800, user-scalable=yes';
        }
        else{
            document.getElementsByTagName('meta')['viewport'].content='width=device-width, initial-scale=1';
        }

        return () => {
            document.getElementsByTagName('meta')['viewport'].content = currentMeta;
        }
    }, []);

    return (
        <div style={{padding: "10px 0px 0px 0px", overflow:"hidden", width: '100%'}}>
            <ProfileIntro isMobile={isMobile} />
            <ProfileSkills />
            <ProfileProjects isMobile={false}  />
        </div>
    )
}
