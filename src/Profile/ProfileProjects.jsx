import React from 'react'
import ProjectCard from './ProjectCard'
import { GiMeshNetwork } from "react-icons/gi";
import { CiViewTable } from "react-icons/ci";
import { SiTensorflow, SiHuggingface, SiMlflow, SiPrometheus, SiGrafana, SiKubernetes, SiNumpy, SiGraphql, SiCplusplus, SiNvidia, SiJupyter, SiScipy, SiScikitlearn, SiNeo4J, SiPytorch, SiCss3, SiFastapi, SiHtml5, SiMongodb, SiMysql, SiPython, SiReact, SiRedux } from "react-icons/si";
import { GrMysql } from "react-icons/gr";
import { DiSqllite } from "react-icons/di";
import { FaGamepad, FaDocker, FaAws } from "react-icons/fa";
import { MdOutlineSoupKitchen } from "react-icons/md";
import Timeline from '@mui/lab/Timeline';
import { binaryTreeGithubURL, bstVisulizationURL, heapVisulizationURL, mathsyraBackendURL, mathsyraFrontendURL } from '../constants';


export default function ProfileProjects({isMobile}) {
    const imagesDashboardData = {
        title: "Images Dashboard",
        id: "images-dashboard",
        oppositeContent: (<>Continuous Development <br /> Industry Project</>),
    
        items: [
            {
                type: "text",
                content: "Images Dashboard is a CCTV image capturing and auditing platform with integration of AI to detect common objects."
            },
            {
                type: "text",
                content: (
                    <div>
                        I developed this during my internship at Integrated active monitoring Pvt Ltd (
                        <a href="https://smartiam.in/" target="_blank">IAM</a>)
                    </div>
                ),
            },
            {
                type: "text",
                content: (<i>( Product related Blog, Video and other informational content to be added soon. )</i>)
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "images-dashboard-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "Designing of the frontend code architecture (which includes setting up the project and donfiguring advanced state management and routing).",
                            "Development and maintainence of the codebase.",
                            "Ticketing software backend architecture and api integration",
                        ],
                    },
                ],
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "React", icon: <SiReact className="timeline-tech-icon" /> },
                    { text: "Redux", icon: <SiRedux className="timeline-tech-icon" /> },
                    { text: "FastAPI", icon: <SiFastapi className="timeline-tech-icon" /> },
                    { text: "MySQL", icon: <GrMysql className="timeline-tech-icon" /> },
                    { text: "MongoDB", icon: <SiMongodb className="timeline-tech-icon" /> },
                ]
            },
        ]
    }

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

    const bayesianCNNData = {
        title: "Bayesian CNN",
        id: "bayesian-cnn",
        oppositeContent: (<>Academic Research <br /> Paper Submission to IJCNN 2025</>),
    
        items: [
            {
                type: "text",
                content: "Researched and developed a novel Bayesian Convolutional Neural Network (BCNN) integrating Gaussian Process priors on weights and utilizing Variational Inference for superior uncertainty quantification."
            },
            {
                type: "text",
                content: "Conducted extensive experimentation to assess predictive uncertainty, model robustness, and calibration performance across various benchmarks."
            },
            {
                type: "text",
                content: "Submitted results to the International Joint Conference on Neural Networks (IJCNN), 2025."
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "bayesian-cnn-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "Designing and implementing Bayesian Convolutional Neural Network architecture with Gaussian Process priors.",
                            "Utilizing Variational Inference techniques for efficient approximation and training.",
                            "Performing uncertainty quantification and calibration analysis.",
                            "Running controlled experiments to evaluate robustness and model reliability.",
                            "Preparing and submitting results to IJCNN 2025."
                        ],
                    },
                ],
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "PyTorch", icon: <SiPytorch className="timeline-tech-icon" /> },
                    { text: "NumPy", icon: <SiNumpy className="timeline-tech-icon" /> },
                    { text: "SciPy", icon: <SiScipy className="timeline-tech-icon" /> },
                    { text: "Scikit-learn", icon: <SiScikitlearn className="timeline-tech-icon" /> },
                ]
            },
        ]
    }

    const rlBoardGamesData = {
        title: "Reinforcement Learning for Board Games",
        id: "rl-board-games",
        oppositeContent: (<>Personal Project <br /> Research & Experimentation</>),
    
        items: [
            {
                type: "text",
                content: "Conceptualized and implemented a Reinforcement Learning (RL) agent for strategic board games such as Catan and Monopoly, enabling autonomous decision-making and adaptive strategy formation."
            },
            {
                type: "text",
                content: "Trained Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents on thousands of simulated gameplays to optimize performance."
            },
            {
                type: "text",
                content: "Applied reward shaping techniques and policy gradient methods to enhance learning efficiency and strategic depth."
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "rl-board-games-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "Designing the environment and state-action spaces for board games.",
                            "Implementing and training RL agents using DQN and PPO algorithms.",
                            "Optimizing agent performance through reward shaping and policy gradient tuning.",
                            "Running large-scale simulations and analyzing agent behavior."
                        ],
                    },
                ],
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "PyTorch", icon: <SiPytorch className="timeline-tech-icon" /> },
                    { text: "TensorFlow", icon: <SiTensorflow className="timeline-tech-icon" /> },
                    { text: "NumPy", icon: <SiNumpy className="timeline-tech-icon" /> },
                    { text: "OpenAI Gym", icon: <FaGamepad className="timeline-tech-icon" /> }, // used OpenCV icon as Gym doesn’t have one
                ]
            },
        ]
    }

    const bayesianOptimizationData = {
        title: "Bayesian Optimization for Seq2Seq Models",
        id: "bayesian-optimization-rnns-transformers",
        oppositeContent: (<>Academic Research and Machine Learning Engineering <br /> Performance Optimization</>),

        items: [
            {
                type: "text",
                content: "Researched and applied Bayesian Optimization techniques to automate hyperparameter tuning for large-scale deep learning models, specifically Recurrent Neural Networks (RNNs) and Transformer architectures containing over 100M parameters."
            },
            {
                type: "text",
                content: "Implemented Gaussian Process-based surrogate models to efficiently explore the hyperparameter space, significantly reducing manual experimentation overhead."
            },
            {
                type: "text",
                content: "Developed custom CUDA (C++) kernels to optimize low-level matrix multiplications and softmax operations critical to Transformer layers, utilizing advanced tiling strategies and shared memory optimization."
            },
            {
                type: "text",
                content: "Achieved a 30% reduction in training time and improved model convergence stability, validated across multiple benchmark datasets including language modeling and time series tasks."
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "bayesian-optimization-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "Designed and implemented Bayesian Optimization pipeline using Gaussian Process surrogates for hyperparameter tuning.",
                            "Integrated Bayesian optimization with PyTorch's training loop, enabling seamless evaluation of multiple hyperparameter configurations.",
                            "Developed custom CUDA kernels for critical matrix operations, optimizing cache coherence and minimizing global memory access.",
                            "Employed tiling and loop unrolling techniques to improve computational throughput.",
                            "Conducted detailed benchmarking and ablation studies to evaluate optimization gains.",
                            "Documented findings and performance improvements for internal publications and research dissemination."
                        ],
                    },
                ],
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "PyTorch", icon: <SiPytorch className="timeline-tech-icon" /> },
                    { text: "C++", icon: <SiCplusplus className="timeline-tech-icon" /> },
                    { text: "CUDA", icon: <SiNvidia className="timeline-tech-icon" /> },
                    { text: "NumPy", icon: <SiNumpy className="timeline-tech-icon" /> },
                    { text: "SciPy", icon: <SiScipy className="timeline-tech-icon" /> },
                    { text: "Jupyter", icon: <SiJupyter className="timeline-tech-icon" /> },
                ]
            },
        ]
    }

    const ragSystemData = {
        title: "RAG",
        id: "rag-narrative-knowledge",
        oppositeContent: (<>Group Project - RAG Exploration <br /> Applied NLP & Knowledge Graphs</>),

        items: [
            {
                type: "text",
                content: "Designed and implemented a Retrieval-Augmented Generation (RAG) system to automate narrative data extraction, character relationship mapping, and consistency checks in large fictional datasets."
            },
            {
                type: "text",
                content: "Integrated Litegraph-based visualization with Large Language Models (LLMs) to dynamically update character knowledge graphs based on extracted contextual information."
            },
            {
                type: "text",
                content: "Employed vector similarity search techniques for efficient context retrieval and relevance ranking, enabling accurate fact extraction and reducing hallucination."
            },
            {
                type: "text",
                content: "Generated tailored datasets to fine-tune LLMs using Low-Rank Adaptation (LoRA), enhancing domain-specific generation quality."
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "rag-narrative-knowledge-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "Architected the RAG system pipeline integrating LLM-based text extraction with vector similarity search.",
                            "Implemented Litegraph-based visualization for real-time character knowledge graph updates.",
                            "Developed FastAPI-based backend for orchestrating data flow, query processing, and vector search operations.",
                            "Utilized MongoDB for structured storage of character relationships and narrative metadata.",
                            "Prepared high-quality LoRA fine-tuning datasets based on extracted and validated knowledge graphs.",
                            "Implemented consistency validation logic to detect contradictions and narrative errors across documents."
                        ],
                    },
                ],
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "PyTorch", icon: <SiPytorch className="timeline-tech-icon" /> },
                    { text: "FastAPI", icon: <SiFastapi className="timeline-tech-icon" /> },
                    { text: "MongoDB", icon: <SiMongodb className="timeline-tech-icon" /> },
                    { text: "NumPy", icon: <SiNumpy className="timeline-tech-icon" /> },
                    { text: "GraphQL", icon: <SiGraphql className="timeline-tech-icon" /> },
                ]
            },
        ]
    }

    const mlopsMovieRecommendationData = {
        title: "MLOps - Cloud Native Recommendor System",
        id: "mlops-movie-recommendation",
        oppositeContent: (<>Academic Project <br /> MLOps & Systems Engineering</>),

        items: [
            {
                type: "text",
                content: "Engineered a scalable, cloud-native movie recommendation system with a full-fledged MLOps pipeline to automate model development, deployment, and monitoring."
            },
            {
                type: "text",
                content: "Utilized MLFlow for experiment tracking, artifact management, and model versioning, ensuring reproducibility across environments."
            },
            {
                type: "text",
                content: "Implemented distributed hyperparameter tuning using Ray on AWS EC2 clusters, reducing search time by 40% for optimal recommendation models."
            },
            {
                type: "text",
                content: "Containerized the application using Docker and deployed it on AWS EKS (Elastic Kubernetes Service), enabling auto-scaling to manage dynamic traffic patterns efficiently."
            },
            {
                type: "text",
                content: "Integrated Prometheus and Grafana dashboards for real-time monitoring of model latency, system health, and user interaction metrics."
            },
            {
                type: "text",
                content: "Processed over 500K+ user interactions daily, with the system dynamically scaling to handle 5x traffic spikes while maintaining sub-100ms response times."
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "mlops-movie-recommendation-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "Designed an end-to-end MLOps pipeline for recommendation system development and deployment.",
                            "Implemented experiment tracking and model management with MLFlow.",
                            "Set up Ray on AWS for distributed hyperparameter optimization, parallelizing training across multiple EC2 instances.",
                            "Developed Docker containers and orchestrated deployments on AWS EKS using Kubernetes.",
                            "Configured Prometheus for metric collection and Grafana for visualization and alerting.",
                            "Optimized system latency and scalability, achieving sub-100ms response times under peak loads."
                        ],
                    },
                ],
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "PyTorch", icon: <SiPytorch className="timeline-tech-icon" /> },
                    { text: "Docker", icon: <FaDocker className="timeline-tech-icon" /> },
                    { text: "Kubernetes", icon: <SiKubernetes className="timeline-tech-icon" /> },
                    { text: "AWS", icon: <FaAws className="timeline-tech-icon" /> },
                    { text: "MLFlow", icon: <SiMlflow className="timeline-tech-icon" /> },
                    { text: "Prometheus", icon: <SiPrometheus className="timeline-tech-icon" /> },
                    { text: "Grafana", icon: <SiGrafana className="timeline-tech-icon" /> },
                ]
            },
        ]
    }

    const pinnFnoData = {
        title: "Physics-Informed Neural Networks for Fluid Dynamics",
        id: "pinn-fluid-dynamics",
        oppositeContent: (<>Academic Research <br /> Scientific Machine Learning</>),
    
        items: [
            {
                type: "text",
                content: "Implemented Physics-Informed Neural Networks (PINNs) to solve Partial Differential Equations (PDEs) in fluid dynamics, enforcing physical laws directly into the neural network training process."
            },
            {
                type: "text",
                content: "Utilized Fourier Neural Operators (FNOs) to approximate PDE solutions efficiently, significantly reducing computational costs and simulation time."
            },
            {
                type: "text",
                content: "Validated model accuracy against traditional numerical solvers, achieving a reduction in computational expense by 10% while maintaining high fidelity to physical constraints."
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "pinn-fluid-dynamics-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "Designed PINN architectures incorporating boundary and initial conditions of PDEs relevant to fluid dynamics.",
                            "Implemented loss functions combining data-driven terms with physics-based PDE residuals.",
                            "Integrated Fourier Neural Operators (FNOs) to capture global dependencies and improve scalability.",
                            "Benchmarked performance against conventional solvers (e.g., Finite Difference Method) for accuracy and efficiency comparison.",
                            "Conducted hyperparameter tuning and stability analysis for convergence improvements."
                        ],
                    },
                ],
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "PyTorch", icon: <SiPytorch className="timeline-tech-icon" /> },
                    { text: "NumPy", icon: <SiNumpy className="timeline-tech-icon" /> },
                    { text: "SciPy", icon: <SiScipy className="timeline-tech-icon" /> },
                    { text: "Jupyter", icon: <SiJupyter className="timeline-tech-icon" /> },
                ]
            },
        ]
    }

    const rnnLstmTradingData = {
        title: "Alpha Generation using Deep RNNs and LSTMs",
        id: "rnn-lstm-trading-strategies",
        oppositeContent: (<>Independent Project <br /> Quantitative Finance</>),
    
        items: [
            {
                type: "text",
                content: "Developed and deployed Deep Recurrent Neural Networks (RNNs) and Long Short-Term Memory Networks (LSTMs) to model and analyze non-stationary financial indicators."
            },
            {
                type: "text",
                content: "Engineered a pipeline to preprocess time series data from NIFTY50 and S&P500 indices, applying normalization, stationarity checks, and feature selection techniques."
            },
            {
                type: "text",
                content: "Designed and trained models to predict short-term price movements, integrating predictions into rule-based trading strategies."
            },
            {
                type: "text",
                content: "Achieved a 4% improvement in cumulative returns over baseline strategies, validated through backtesting across multiple market conditions."
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "rnn-lstm-trading-strategies-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "Collected and preprocessed historical stock price data and technical indicators for NIFTY50 and S&P500 indices.",
                            "Implemented Deep RNN and LSTM architectures for capturing temporal dependencies in financial time series.",
                            "Applied rolling window and walk-forward validation techniques for robust model evaluation.",
                            "Integrated model predictions into systematic trading strategies with position sizing and risk management rules.",
                            "Performed performance evaluation through backtesting, including Sharpe ratio, drawdown, and cumulative return analysis."
                        ],
                    },
                ],
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "PyTorch", icon: <SiPytorch className="timeline-tech-icon" /> },
                    { text: "NumPy", icon: <SiNumpy className="timeline-tech-icon" /> },
                    { text: "Pandas", icon: <CiViewTable className="timeline-tech-icon" /> },
                    { text: "Scikit-learn", icon: <SiScikitlearn className="timeline-tech-icon" /> },
                    { text: "Jupyter", icon: <SiJupyter className="timeline-tech-icon" /> },
                ]
            },
        ]
    }

    const llmSummarizationData = {
        title: "LLM-based Research Paper Summarization System",
        id: "llm-research-summarization",
        oppositeContent: (<>Academic Tool <br /> NLP Workflow Automation</>),
    
        items: [
            {
                type: "text",
                content: "Engineered automated workflows leveraging Large Language Models (LLMs) to streamline research paper summarization and literature review processes for NYU faculty and students."
            },
            {
                type: "text",
                content: "Integrated LangChain to orchestrate multi-step LLM-based pipelines, combining paper ingestion, metadata extraction, content chunking, and summarization steps seamlessly."
            },
            {
                type: "text",
                content: "Utilized HuggingFace pre-trained transformer models fine-tuned for academic and scientific text, ensuring accurate content extraction and domain-specific summarization."
            },
            {
                type: "text",
                content: "Developed a FastAPI-based interface allowing users to upload papers (PDFs), retrieve key insights, and generate structured literature reviews with citation metadata."
            },
            {
                type: "text",
                content: "Reduced manual effort in literature review workflows by 25%, improving consistency and accuracy in information extraction across diverse research domains."
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "llm-research-summarization-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "Designed end-to-end LLM-powered workflows for research paper summarization and review tasks.",
                            "Integrated LangChain to orchestrate parsing, chunking, summarization, and citation extraction steps.",
                            "Utilized HuggingFace transformer models fine-tuned for scientific document processing.",
                            "Implemented a FastAPI interface for user interaction and API deployment.",
                            "Optimized chunking strategies and prompt templates to improve summarization quality and factual consistency.",
                            "Evaluated system performance through manual cross-checking and user feedback loops at NYU."
                        ],
                    },
                ],
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "PyTorch", icon: <SiPytorch className="timeline-tech-icon" /> },
                    { text: "HuggingFace", icon: <SiHuggingface className="timeline-tech-icon" /> },
                    { text: "LangChain", icon: <SiHuggingface className="timeline-tech-icon" /> }, // Since LangChain doesn't have its own icon, using HuggingFace visually
                    { text: "FastAPI", icon: <SiFastapi className="timeline-tech-icon" /> },
                    { text: "NumPy", icon: <SiNumpy className="timeline-tech-icon" /> },
                    { text: "Jupyter", icon: <SiJupyter className="timeline-tech-icon" /> },
                ]
            },
        ]
    }

    
    const mathsyraData = {
        title: "Mathsyra",
        id: "mathsyra",
        oppositeContent: "April 2021",
    
        items: [
            {
                type: "text",
                content: "Mathsyra is a web application for students to learn the concepts of mathematics in an innovative way with Indian cultural themed UI and quizzes."
            },
            {
                type: "collapse",
                title: "Objective",
                id: "mathsyra-objective",
                items: [
                    {
                        type: "list",
                        content: [
                            "This project was created for Toycathon 2021."
                        ]
                    }
                ]
            },
            {
                type: "collapse",
                title: "Key Features",
                id: "mathsyra-key-features",
                items: [
                    {
                        type: "list",
                        content: [
                            "Well Designed Interface and User Experience",
                            "Perfectly categorized modules and quizzes.",
                            "Interactive Blogs related to vedic maths to Understand the concepts and applications.",
                            "Quiz after every module to test your knowledge."
                        ]
                    },
                ]
            },
            {
                type: "collapse",
                title: "Achievements",
                id: "mathsyra-achievements",
                items: [
                    {
                        type: "list",
                        content: [
                            "Finalist in Toycathon 2021."
                        ]
                    },
                ]
            },
            {
                type: "chips",
                title: "Tech/Learn Stack:",
                content: [
                    { text: "FastAPI", icon: <SiFastapi className="timeline-tech-icon" /> },
                    { text: "React", icon: <SiReact className="timeline-tech-icon" /> },
                    { text: "SQL lite", icon: <DiSqllite className="timeline-tech-icon" /> },
                ]
            },
            {
                type: "links",
                content: [
                    { text: "Frontend", link: mathsyraFrontendURL },
                    { text: "Backend", link: mathsyraBackendURL },
                ]
            }    
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
        bayesianCNNData,
        llmSummarizationData,
        rlBoardGamesData,
        bayesianOptimizationData,
        ragSystemData,
        mlopsMovieRecommendationData,
        pinnFnoData,
        rnnLstmTradingData,
        imagesDashboardData,
        binaryTreeData,
        mathsyraData,
    ]

    // useEffect(() => {
    //     alert("is mobile: " + isMobile);
    //     // console.log("is mobile: " + isMobile);
    // }, [isMobile])

    return (
        <div style={{marginTop: '80px', textAlign: 'center'}} data-aos="fade-up" data-aos-once="true" >
            <div className="profile-title-div" style={{textAlign: 'center', lineHeight: '1'}} >
                <span className='profile-subtitle-text' style={{fontSize: '35px'}}>
                    Projects
                </span>
            </div>
            <div className="profile-title-div" style={{textAlign: 'center'}} >
                <span className="profile-description-div" style={{marginTop: "0px", fontSize: '15px'}}>
                    My projects ordered by complexity
                </span>
            </div>
            <Timeline position="alternate">
                <div style={{aliginItems: 'center'}}>
                    <div style={{ display: 'inline-block', borderBottom: '2px solid #fff', minWidth: '400px'}}></div>    
                </div>

                {
                    projects.map((project, index) => {
                        return (
                            <ProjectCard data={project} align={index % 2 === 0 ? "left" : "right"} 
                                aosAnimation={
                                    isMobile ? "fade-up" : index % 2 === 0 ? "zoom-out-left" : "zoom-out-right"
                                } 
                            />
                        )
                    })
                }

            </Timeline>

        </div>
    )
}
