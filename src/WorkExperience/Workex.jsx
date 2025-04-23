import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './Workex.css';
import ProjectCard from '../Profile/ProjectCard'
import { Link } from 'react-router-dom';
import { CiViewTable } from "react-icons/ci";
import { SiHuggingface, SiPlotly, SiMlflow, SiPrometheus, SiGrafana, SiKubernetes, SiNumpy, SiGraphql, SiCplusplus, SiNvidia, SiJupyter, SiOpencv, SiScikitlearn, SiPytorch, SiCss3, SiFastapi, SiHtml5, SiMongodb, SiMysql, SiPython, SiReact } from "react-icons/si";
import { GrMysql } from "react-icons/gr";
import { FaDocker, FaAws } from "react-icons/fa";
import Timeline from '@mui/lab/Timeline';
import { LuBrainCircuit } from "react-icons/lu";

export default function Workex({isMobile}) {
    document.title = "Yash Darak | Work Experience";
    const nyuworkex = {
        title: "Research Assistant/Machine Learning Engineer",
        id: "nyu-workex",
        oppositeContent: (<>New York University<br /> Sep 2023 - Present</>),
    
        items: [
            {
                type: "text",
                content: "Researched and implemented Bayesian Optimization to tune Deep Neural Networks (RNNs, Transformers), improving training stability and reducing inference variance by 28% across 100M+ parameter models using PyTorch."
            },
            {
                type: "text",
                content: ""
            },
            {
                type: "text",
                content: "Optimized high performance CUDA(C++) kernels for large-scale training pipelines, reducing runtime by 30% in distributed GPU environments."
            },
            {
                type: "text",
                content: ""
            },
            {
                type: "text",
                content: "Fine-tuned and pre-trained Large Language Models (LLMs) using Langchain and HuggingFace to automate research paper summarization and literature review tasks, \
                            reducing document review time. Optimized model performance using techniques like quantization and distillation for deployment on constrained environments, \
                            reducing manual effort by 25% and improving content extraction accuracy for faculty and students at NYU. involved converting raw financial data into structured \
                            knowledge graphs using Neo4j, enabling complex relationship modeling."
            },
            {
                type: "text",
                content: ""
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "Pytorch", icon: <SiPytorch className="timeline-tech-icon" /> },
                    { text: "C++", icon: <SiCplusplus className="timeline-tech-icon" /> },
                    { text: "CUDA", icon: <SiNvidia className="timeline-tech-icon" /> },
                    { text: "HuggingFace", icon: <SiHuggingface className="timeline-tech-icon" /> },
                ]
            },
        ]
    }
         
    const iamworkex2 = {
        title: "Machine Learning Engineer",
        id: "iam-workex2",
        oppositeContent: (<>IAM. Pvt. Ltd.<br /> Oct 2022 - Jun 2023</>),

        items: [
            {
                type: "text",
                content: "Designed and trained convolutional neural network models for motion detection and face recognition using OpenCV, PyTorch, and custom data augmentation pipelines—achieved 92% accuracy on a 10K image test set while sustaining 25 FPS inference."
            },
            {
                type: "text",
                content: ""
            },
            {
                type: "text",
                content: "Integrated the trained models into a real time CCTV analytics pipeline, reducing false positives by 40% and enabling 24/7 automated alerting."
            },
            {
                type: "text",
                content: ""
            },
            {
                type: "text",
                content: "Engineered ML pipelines using PySpark, Airflow, and S3, processing 10GB+ of retail (transactional, footfall, energy) data daily for facilities exceeding 20,000 sq.ft to support KPI dashboards and real-time ops decisions.",
            },
            {
                type: "text",
                content: ""
            },
            {
                type: "text",
                content: "Engineered robust features from time-series RFID datasets and implemented Multivariate Regression models to improve forecasting accuracy by 20%, using ROC-AUC, RMSE, and confusion matrix for model assessment and validation."
            },
            {
                type: "text",
                content: ""
            },
            {
                type: "text",
                content: "Streamlined SQL queries (PostgreSQL) to accelerate production data extraction, cutting report generation time by 20% in Power BI dashboards.",
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "Deep Learning", icon: <LuBrainCircuit className="timeline-tech-icon" /> },
                    { text: "React", icon: <SiReact className="timeline-tech-icon" /> },
                    { text: "OpenCV", icon: <SiOpencv className="timeline-tech-icon" /> },
                    { text: "FastAPI", icon: <SiFastapi className="timeline-tech-icon" /> },
                    { text: "PostgreSQL", icon: <GrMysql className="timeline-tech-icon" /> },
                    { text: "PySpark", icon: <SiNumpy className="timeline-tech-icon" /> },
                    { text: "MongoDB", icon: <SiMongodb className="timeline-tech-icon" /> },
                    { text: "S3", icon: <FaAws className="timeline-tech-icon" /> },
                    { text: "SQL", icon: <SiMysql className="timeline-tech-icon" /> },
                ]
            }
        ]
    }

    const atlascopcoworkex = {
        title: "Machine Learning Engineer",
        id: "atlascopco-workex", 
        oppositeContent: (<>Atlas Copco<br /> Aug 2021 - Sep 2022</>),

        items: [
            {
                type: "text",
                content: "Built predictive models and anomaly detection systems from unstructured IoT sensor data using clustering and Statistical Learning, enabling early failure detection \
                           and reducing operational cost by $50K per year. Validated models by cross-validation and residual error analysis. and trained convolutional neural network models for motion \
                           detection and face recognition using OpenCV, PyTorch, and custom data augmentation pipelines—achieved 92% accuracy on a 10K image test set while sustaining 25 FPS inference."
            },
            {
                type: "text",
                content: ""
            },
            {
                type: "text",
                content: "Designed robust CI/CD pipelines (Jenkins, GitLab CI) to streamline A/B testing and iterative deployment of ML models into production, reducing testing turnaround by 20%, \
                          directly supporting experimentation culture. the trained models into a real time CCTV analytics pipeline, reducing false positives by 40% and enabling 24/7 automated alerting."
            },
            {
                type: "text",
                content: ""
            },
            {
                type: "text",
                content: "Engineered and deployed low-latency deep learning inference APIs (<20ms) using FastAPI, Docker, and Kubernetes. Implemented quantization and caching strategies to reduce \
                          latency by 25%.  ML pipelines using PySpark, Airflow, and S3, processing 10GB+ of retail (transactional, footfall, energy) data daily for facilities exceeding 20,000 sq.ft \
                          to support KPI dashboards and real-time ops decisions.",
            },
            {
                type: "text",
                content: ""
            },
            {
                type: "text",
                content: "Built CI/CD pipelines enabling <1-hour rollbacks and 99.9% uptime for real-time industrial monitoring services.d robust features from time-series RFID datasets and implemented Multivariate Regression models to improve forecasting accuracy by 20%, using ROC-AUC, RMSE, and confusion matrix for model assessment and validation."
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "Deep Learning", icon: <LuBrainCircuit className="timeline-tech-icon" /> },
                    { text: "PowerBI", icon: <CiViewTable className="timeline-tech-icon" /> },
                    { text: "FastAPI", icon: <SiFastapi className="timeline-tech-icon" /> },
                    { text: "PostgreSQL", icon: <GrMysql className="timeline-tech-icon" /> },
                    { text: "PySpark", icon: <SiNumpy className="timeline-tech-icon" /> },
                    { text: "Docker", icon: <FaDocker className="timeline-tech-icon" /> },
                    { text: "Kubernetes", icon: <SiKubernetes className="timeline-tech-icon" /> },
                    { text: "Grafana", icon: <SiGrafana className="timeline-tech-icon" /> },
                    { text: "Prometheus", icon: <SiPrometheus className="timeline-tech-icon" /> },
                    { text: "MLFlow", icon: <SiMlflow className="timeline-tech-icon" /> },
                ]
            }
        ]
    }

    const iamworkex1 = {
        title: "Machine Learning Intern",
        id: "iam-workex1", 
        oppositeContent: (<>IAM Pvt. Ltd.<br /> Jan 2021 - Jun 2021</>),

        items: [
            {
                type: "text",
                content: "Implemented experiment tracking via MLFlow, using confidence intervals and uplift metrics to compare alternate routing models for cloud kitchen logistics."
            },
            {
                type: "text",
                content: ""
            },
            {
                type: "text",
                content: "Worked independently with business stakeholders to scope metrics for optimization (delivery time, cost) and reduce logistics overhead by 20%."
            },
            {
                type: "text",
                content: ""
            },
            {
                type: "text",
                content: "•	Automated model deployment across dev/staging/prod environments using Ansible and GitLab CI, ensuring reproducibility and continuous integration.",
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "MLFlow", icon: <SiMlflow className="timeline-tech-icon" /> },
                    { text: "Ansible", icon: <SiNumpy className="timeline-tech-icon" /> },
                    { text: "PostgreSQL", icon: <GrMysql className="timeline-tech-icon" /> },
                ]
            }
        ]
    }

    const vitworkex = {
        title: "Undergraduate Research Assistant (CFD)",
        id: "vit-workex", 
        oppositeContent: (<>Vishwakarma Institute of Technology<br /> Jan 2020 - Dec 2020</>),

        items: [
            {
                type: "text",
                content: "Conducted Data Analysis using Pandas and Matplotlib for 50+ peer-reviewed research papers and general help with research experiment tracking via MLFlow, using confidence intervals and uplift metrics to compare alternate routing models for cloud kitchen logistics."
            },
            {
                type: "text",
                content: ""
            },
            {
                type: "text",
                content: "Defined Mathematical Models for fluid flow and pressure analysis using MATLAB for integrity assessments and failure analysis. independently with business stakeholders to scope metrics for optimization (delivery time, cost) and reduce logistics overhead by 20%."
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "Matplotlib", icon: <SiPlotly className="timeline-tech-icon" /> },
                    { text: "R", icon: <SiNumpy className="timeline-tech-icon" /> },
                    { text: "Pandas", icon: <CiViewTable className="timeline-tech-icon" /> },
                    { text: "MATLAB", icon: <SiNumpy className="timeline-tech-icon" /> },
                ]
            }
        ]
    }

    const projects = [
        nyuworkex,
        iamworkex2,
        atlascopcoworkex,
        iamworkex1,
        vitworkex

    ]

    return (
        <div style={{marginTop: '80px', textAlign: 'center'}} data-aos="fade-up" data-aos-once="true" >
            <div className="profile-title-div" style={{textAlign: 'center', lineHeight: '1'}} >
                <span className='profile-subtitle-text' style={{fontSize: '35px'}}>
                    Work Experience
                </span>
            </div>
            <div className="profile-title-div" style={{textAlign: 'center'}} >
                <span className="profile-description-div" style={{marginTop: "0px", fontSize: '16px'}}>
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