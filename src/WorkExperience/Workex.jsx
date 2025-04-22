import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './Workex.css';
import ProjectCard from '../Profile/ProjectCard'
import { Link } from 'react-router-dom';
import { FaArrowLeft } from 'react-icons/fa';
import { GiMeshNetwork } from "react-icons/gi";
import { CiViewTable } from "react-icons/ci";
import { SiTensorflow, SiHuggingface, SiMlflow, SiPrometheus, SiGrafana, SiKubernetes, SiNumpy, SiGraphql, SiCplusplus, SiNvidia, SiJupyter, SiScipy, SiScikitlearn, SiNeo4J, SiPytorch, SiCss3, SiFastapi, SiHtml5, SiMongodb, SiMysql, SiPython, SiReact, SiRedux } from "react-icons/si";
import { GrMysql } from "react-icons/gr";
import { DiSqllite } from "react-icons/di";
import { FaGamepad, FaDocker, FaAws } from "react-icons/fa";
import { MdOutlineSoupKitchen } from "react-icons/md";
import Timeline from '@mui/lab/Timeline';
import { binaryTreeGithubURL, bstVisulizationURL, heapVisulizationURL, mathsyraBackendURL, mathsyraFrontendURL } from '../constants';

export default function Workex({isMobile}) {
    document.title = "Yash Darak | Work Experience";
    const graphDLData = {
        title: "Graph Deep Learning (GNNs)",
        id: "graph-deep-learning-sparse",
        oppositeContent: (<>Academic Research with a Professor<br /> Currently writing the paper!</>),
    
        items: [
            {
                type: "text",
                content: "I proposed and developed a Graph Machine Learning model aimed at improving predictions of potential mergers and acquisitions (M&A)."
            },
            {
                type: "text",
                content: "I led the end-to-end pipeline, starting from extracting and processing over 250 GB of online financial data through web scraping techniques."
            },
            {
                type: "text",
                content: "This involved converting raw financial data into structured knowledge graphs using Neo4j, enabling complex relationship modeling."
            },
            {
                type: "text",
                content: "The most exciting part was implementing predictive algorithms for sparse M&A networks, utilizing Graph Neural Networks (GNNs) and node embeddings to identify acquisition patterns and hidden relationships."
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "graph-deep-learning-sparse-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "End-to-end project leadership: from data extraction to model deployment.",
                            "Web scraping and cleaning over 250 GB of financial datasets.",
                            "Knowledge graph design and implementation using Neo4j.",
                            "Modeling acquisition relationships using Graph Neural Networks (GNNs).",
                            "Training and optimizing models for sparse graph data."
                        ],
                    },
                ],
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "Pytorch", icon: <SiPytorch className="timeline-tech-icon" /> },
                    { text: "Neo4j", icon: <SiNeo4J className="timeline-tech-icon" /> },
                    { text: "PyTorch Geometric", icon: <SiPytorch className="timeline-tech-icon" /> },
                    { text: "BeautifulSoup", icon: <MdOutlineSoupKitchen className="timeline-tech-icon" /> },
                    { text: "NetworkX", icon: <GiMeshNetwork className="timeline-tech-icon" /> },
                ]
            },
        ]
    }
         
    const binaryTreeData = {
        title: "Binary Tree Visualizer",
        id: "binary-tree",
        oppositeContent: "March 2021",

        items: [
            {
                type: "text",
                content: "An application to visualize insertion, deletion, search and structure of nodes in a Binary Search Tree (BST) and Heap.",
            },
            {
                type: "text",
                content: "The nodes and edges of binary tree are highlighted to depict the path of comparisons that took place to insert/search a node.",
            },
            {
                type: "text",
                content: "This could be very helpful for students to understand the working of BST and Heap.",
            },
            {
                type: "chips",
                title: "Tech/Learn Stack",
                content: [
                    { text: "React", icon: <SiReact className="timeline-tech-icon" /> },
                    { text: "Html", icon: <SiHtml5 className="timeline-tech-icon" /> },
                    { text: "CSS", icon: <SiCss3 className="timeline-tech-icon" /> },
                ]
            },
            {
                type: "links",
                content: [
                    { text: "Github", link: binaryTreeGithubURL },
                    { text: "Heap", link: heapVisulizationURL },
                    { text: "Binary Search Tree", link: bstVisulizationURL },
                ]
            }
        ]
    }


    const projects = [
        graphDLData,
        binaryTreeData,
    ]

    return (
        <div style={{marginTop: '80px', textAlign: 'center'}} data-aos="fade-up" data-aos-once="true" >
            <div className="profile-title-div" style={{textAlign: 'center', lineHeight: '1'}} >
                <span className='profile-subtitle-text' style={{fontSize: '35px'}}>
                    Work Experience
                </span>
            </div>
            <div className="profile-title-div" style={{textAlign: 'center'}} >
                <span className="profile-description-div" style={{marginTop: "0px", fontSize: '15px'}}>
                    My humble attempt to contribute to the world of technology and machine learning.
                </span>
            </div>
            <Timeline position="alternate">
                <div style={{aliginItems: 'center'}}>
                    <div style={{ display: 'inline-block', borderBottom: '2px solid #fff', minWidth: '400px'}}></div>    
                </div>

                {
                    projects.map((project, index) => {
                        return (
                            <ProjectCard 
                                key={project.id || `project-${index}`}
                                data={project} align={index % 2 === 0 ? "left" : "right"} 
                                aosAnimation={ isMobile ? "fade-up" : index % 2 === 0 ? "zoom-out-left" : "zoom-out-right"} 
                            />
                        )
                    })
                }

            </Timeline>

        </div>
    )
}