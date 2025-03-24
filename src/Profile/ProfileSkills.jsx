import React from 'react'
import { SiCplusplus, SiScikitlearn, SiCss3, SiDocker, SiNumpy, SiFastapi, SiPytorch, SiFlask, SiGit, SiHtml5, SiJavascript, SiJson, SiMysql, SiPython, SiReact } from "react-icons/si";
import { CiViewTable } from "react-icons/ci";
import { GiBrain } from "react-icons/gi";
import { AiOutlineSolution, AiOutlineTeam, AiFillFileExcel } from "react-icons/ai";
import { GrMysql } from "react-icons/gr";
import { MdOutlineBakeryDining } from "react-icons/md";

export default function ProfileSkills() {

    const ref = React.useRef(null);
    // useEffect(() => {
    //     if (isInViewport && window.location.hash !== `#skills`) {
    //         // setCollapseKeys(defaultExpanded);
    //         // push the #id to the url
    //         // replace
    //         window.history.replaceState(null, null, `#skills`);
    //     }
    // }, [isInViewport]);

    const skills = [
        { name: 'Python', icon: <SiPython className="profile-tech-icon" /> },
        { name: 'Pytorch', icon: <SiPytorch className="profile-tech-icon" /> },
        { name: 'Machine Learning', icon: <GiBrain className="profile-tech-icon" /> },
        { name: 'Scikit-Learn', icon: <SiScikitlearn className="profile-tech-icon" /> },
        { name: 'NumPy', icon: <SiNumpy className="profile-tech-icon" /> },
        { name: 'Pandas', icon: <CiViewTable className="profile-tech-icon" /> },
        { name: 'Excel', icon: <AiFillFileExcel className="profile-tech-icon" /> },
        { name: 'C++', icon: <SiCplusplus className="profile-tech-icon" /> },
        { name: 'SQL', icon: <GrMysql className="profile-tech-icon" /> },
        { name: 'Docker', icon: <SiDocker className="profile-tech-icon" /> },
        { name: 'React', icon: <SiReact className="profile-tech-icon" /> },
        { name: 'Javascript', icon: <SiJavascript className="profile-tech-icon" /> },
        { name: 'Html', icon: <SiHtml5 className="profile-tech-icon" /> },
        { name: 'Css', icon: <SiCss3 className="profile-tech-icon" /> },
        { name: 'Git', icon: <SiGit className="profile-tech-icon" /> },
        { name: 'FastAPI', icon: <SiFastapi className="profile-tech-icon" /> },
        { name: 'Teamwork', icon: <AiOutlineTeam className="profile-tech-icon" /> },
        { name: 'Problem Solving', icon: <AiOutlineSolution className="profile-tech-icon" /> },
    ]

    return (
        <div style={{marginTop: '60px', textAlign: 'center'}} id="skills" ref={ref} data-aos="fade-up" data-aos-once="true">
            <div className="profile-title-div" style={{textAlign: 'center', lineHeight: '1'}} >
                <span className='profile-subtitle-text' style={{fontSize: '35px'}}>
                    Skills
                </span>
            </div>
            <div className="profile-title-div" style={{textAlign: 'center'}} >
                <span className="profile-description-div" style={{marginTop: "0px", fontSize: '15px'}}>
                    I can say i'm quite good at
                </span>
            </div>
            <div style={{marginTop: '14px', maxWidth: '500px', display: 'inline-flex'}}>
            <div>

                {skills.map((skill, index) => {
                    return (
                        <span className="profile-skills-chip" key={index} 
                            data-aos="fade-up" data-aos-delay={index * 100 + 100} data-aos-once="true" data-aos-anchor="#skills" 
                        >
                            {skill.icon}
                            {skill.name}
                        </span>
                    )
                }
                )}
                </div>
            </div>
        </div>
    )
}
