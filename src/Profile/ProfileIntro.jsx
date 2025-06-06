import React, { useEffect, useRef, memo } from "react";
import { init } from "ityped";
import { FiLinkedin, FiMail } from 'react-icons/fi';
import { BsWhatsapp } from 'react-icons/bs';
import { Tooltip } from 'antd';
import { FaGithub } from "react-icons/fa";
import { emailId, whatsappNumber, linkedinLink, githubLink } from "../constants";

// Use React.memo to prevent unnecessary re-renders
const ProfileIntro = memo(({ isMobile }) => {
    const textRef = useRef();
    const ref = React.useRef(null);

    useEffect(() => {
        // Typing animation - only run when dependencies change
        if (textRef.current) {
            init(textRef.current, {
                showCursor: true,
                backDelay: 1500,
                backSpeed: 60,
                strings: (isMobile 
                    ? ["Math Geek.", "Sudoku Enthusiast.", "Problem Solver.", "Logician."]
                    : ["Math Geek. Sudoku Enthusiast. Problem Solver. Logician."]),
            });
        }
    }, [textRef, isMobile]);

    return (
        <div className="profile-outer-div" ref={ref} id="intro">
            <div style={{ minWidth: "350px", textAlign: "left" }} data-aos="fade-up-right" data-aos-duration="500">
                <div style={{ fontSize: "35px" }} className="profile-title-text">
                    Hey there, I'm
                </div>
                <div className="profile-title-text" style={{lineHeight: '1'}}>Yash Darak</div>
                <div data-aos="zoom-in" data-aos-delay="300">
                    <span className="profile-subtitle-text" ref={textRef}></span>
                </div>
                <div className="profile-description-div" style={{ marginTop: "20px" }} >
                    I am a curious human. I am currently pursuing my Masters in Applied Mathematics from NYU.
                </div>
                <div className="profile-description-div">
                    I am fascianted by Machine Learning, Data Science and Quantitative Research.
                </div>
                <div className="profile-description-div">
                    I am very proficient in Algorithmic Thinking and shipping code!
                </div>
                <div className="profile-description-div">
                    A few things about me - I am an ABACUS/UCMAS Gold Medallist, a frequent ranker in Indian Sudoku Championships and I love to play football and cricket!
                </div>
                <div className="profile-description-div" style={{ marginTop: "20px", display: "inline-flex" }}>
                    <div data-aos="fade-left" data-aos-delay="100" data-aos-once="true" >
                        <Tooltip title={`${whatsappNumber}`} placement="bottom">
                            <a
                                href={`https://wa.me/${whatsappNumber}`}
                                target="_blank"
                                className="profile-link-btn profile-btn-gradient-border"
                                style={{ marginLeft: "0px" }}
                            >
                                <BsWhatsapp />
                            </a>
                        </Tooltip>
                    </div>
                    <div data-aos="fade-left" data-aos-delay="200" data-aos-once="true" >
                        <Tooltip title={emailId} placement="bottom">
                            <a
                                href={`mailto:${emailId}`}
                                target="_blank"
                                className="profile-link-btn profile-btn-gradient-border"
                                style={{ marginLeft: "0px" }}
                            >
                                <FiMail />
                            </a>
                        </Tooltip>
                    </div>
                    <div data-aos="fade-left" data-aos-delay="300" data-aos-once="true" >
                        <Tooltip title="LinkedIn" placement="bottom">
                            <a
                                href={linkedinLink}
                                target="_blank"
                                className="profile-link-btn profile-btn-gradient-border"
                                style={{ marginLeft: "0px" }}
                            >
                                <FiLinkedin />
                            </a>
                        </Tooltip>
                    </div>
                    <div data-aos="fade-left" data-aos-delay="100" data-aos-once="true" >
                        <Tooltip title="Github" placement="bottom">
                            <a
                                href={githubLink}
                                target="_blank"
                                className="profile-link-btn profile-btn-gradient-border"
                                style={{ marginLeft: "0px" }}
                            >
                                <FaGithub />
                            </a>
                        </Tooltip>
                    </div>
                </div>
            </div>
            <div data-aos="fade-up-left" data-aos-duration="500">
                <img
                    src="https://avatars.githubusercontent.com/u/126472966?s=400&u=0efefeda99114634f60b415af8491f787519c379&v=4"
                    alt="Yash Darak"
                    className="proflie-photo-img"
                    width="300" 
                    height="300"
                />
            </div>
        </div>
    );
});

export default ProfileIntro;