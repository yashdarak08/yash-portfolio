import React from 'react'
import ProjectCard from './ProjectCard'
import { GiMeshNetwork } from "react-icons/gi";
import { CiViewTable } from "react-icons/ci";
import { SiTensorflow, SiMlflow, SiHuggingface, SiTerraform, SiAmazonec2, SiPrometheus, SiGrafana, SiKubernetes, SiNumpy, SiGraphql, SiCplusplus, SiNvidia, SiJupyter, 
         SiScipy, SiScikitlearn, SiNeo4J, SiPytorch, SiCss3, SiFastapi, SiHtml5, SiMongodb, SiMysql, SiPython, SiReact, SiRedux, SiStreamlit } from "react-icons/si";
import { GrMysql } from "react-icons/gr";
import { DiSqllite } from "react-icons/di";
import { FaGamepad, FaDocker, FaAws } from "react-icons/fa";
import { MdOutlineSoupKitchen } from "react-icons/md";
import Timeline from '@mui/lab/Timeline';
import { binaryTreeGithubURL, graphnnGithubURL, interestRatesGithubURL, biAssistantGithubURL, movieRecommenderGithubURL, twitterSentimentGithubURL, 
         boardGameRLGithubURL, quantTradingGithubURL, bayesianCNNGithubURL } from '../constants';



export default function ProfileProjects({isMobile}) {
    
    const electionSentimentData = {
        title: "Twitter Sentiment Analysis (2024 U.S. Election)",
        id: "election-sentiment-nlp",
        oppositeContent: (<>NLP + Financial Signals<br /> Social Media Analytics</>),
    
        items: [
            {
                type: "text",
                content: "Built a sentiment analysis pipeline to analyze 200,000+ tweets during the 2024 U.S. Presidential Election, focusing on public opinion and market response."
            },
            {
                type: "text",
                content: "Utilized VADER and Naive Bayes classifiers to categorize tweets as positive, negative, or neutral, identifying candidate-level sentiment bias."
            },
            {
                type: "text",
                content: "Applied Chi-Square feature selection to improve classification performance and reduce dimensionality of the tweet feature space."
            },
            {
                type: "text",
                content: "Correlated sentiment trends with financial indicators such as the S&P 500 and Russell 2000 to uncover a 30% spike in market volatility during peak sentiment periods."
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "election-sentiment-nlp-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "Collected and preprocessed 200,000+ tweets using Pandas.",
                            "Built sentiment classifiers using VADER and Naive Bayes.",
                            "Used Chi-Square test for feature importance and selection.",
                            "Visualized sentiment dynamics and political candidate bias.",
                            "Performed financial correlation analysis to connect sentiment with volatility metrics."
                        ],
                    },
                ],
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "NLTK", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "Pandas", icon: <CiViewTable className="timeline-tech-icon" /> },
                    { text: "Scikit-learn", icon: <SiScikitlearn className="timeline-tech-icon" /> },
                    { text: "VADER", icon: <SiPython className="timeline-tech-icon" /> }
                ]
            },
            {
                type: "links",
                content: [
                    { text: "Github", link: twitterSentimentGithubURL },
                ]
            }
        ]
    }
    

    const llmBiAssistantData = {
        title: "LLM-powered Business Intelligence Assistant",
        id: "llm-bi-rag-dashboard",
        oppositeContent: (<>Real-time Document Intelligence<br /> Powered by LLMs + RAG</>),
    
        items: [
            {
                type: "text",
                content: "Built a production-grade Business Intelligence Assistant using a Retrieval-Augmented Generation (RAG) pipeline for real-time decision support."
            },
            {
                type: "text",
                content: "Integrated HuggingFace Transformers with FAISS for semantic search and document chunking to enable efficient information retrieval from business files."
            },
            {
                type: "text",
                content: "Deployed APIs via FastAPI for LLM-powered Q&A over enterprise documents, with latency-optimized endpoints."
            },
            {
                type: "text",
                content: "Developed a responsive Streamlit dashboard for interactive querying and visualization of insights."
            },
            {
                type: "text",
                content: "Enabled real-time system health and latency monitoring using Prometheus, ensuring consistent performance."
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "llm-bi-rag-dashboard-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "Designed the full RAG pipeline using FAISS and Transformers.",
                            "Built backend APIs for LLM-powered semantic search using FastAPI.",
                            "Developed front-end Streamlit dashboard for user interaction.",
                            "Deployed Prometheus-based metrics monitoring for real-time usage analysis.",
                            "Optimized the assistant for latency and accuracy tradeoffs in production settings."
                        ],
                    },
                ],
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "FastAPI", icon: <SiFastapi className="timeline-tech-icon" /> },
                    { text: "Transformers", icon: <SiHuggingface className="timeline-tech-icon" /> },
                    { text: "Streamlit", icon: <SiStreamlit className="timeline-tech-icon" /> },
                    { text: "Prometheus", icon: <SiPrometheus className="timeline-tech-icon" /> },
                    { text: "Docker", icon: <FaDocker className="timeline-tech-icon" /> }
                ]
            },
            {
                type: "links",
                content: [
                    { text: "Github", link: biAssistantGithubURL },
                ]
            }
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
            {
                type: "links",
                content: [
                    { text: "Github", link: graphnnGithubURL }
                ]
            }
        ]
    }

    const interestRateModelingData = {
        title: "Quantitative Modelling: Interest Rate Derivatives",
        id: "interest-rate-derivatives-modeling",
        oppositeContent: (<>Advanced Financial Modeling<br /> Academic + Independent Research</>),
    
        items: [
            {
                type: "text",
                content: "Built quantitative models for pricing interest rate derivatives using the Vasicek and CIR stochastic frameworks."
            },
            {
                type: "text",
                content: "Implemented both finite-difference PDE solvers and Monte Carlo simulations (10,000+ paths) to evaluate derivative prices and compute Value at Risk (VaR) and Conditional VaR (CVaR)."
            },
            {
                type: "text",
                content: "Conducted rigorous statistical validation using diagnostic tests such as Jarque-Bera, Ljung-Box, and Levene’s test to assess model robustness."
            },
            {
                type: "text",
                content: "Performed stress testing to simulate extreme rate shift scenarios and analyze tail risk exposures under stochastic interest rate conditions."
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "interest-rate-derivatives-modeling-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "Implemented Vasicek and CIR short-rate models for bond and option pricing.",
                            "Applied Monte Carlo methods for probabilistic risk estimation.",
                            "Developed finite-difference solvers for PDE-based pricing approaches.",
                            "Back-tested model performance with historical rate data.",
                            "Validated assumptions with econometric statistical tests."
                        ],
                    },
                ],
            },
            {
                type: "chips",
                title: "Tech Stack",
                content: [
                    { text: "Python", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "NumPy", icon: <SiNumpy className="timeline-tech-icon" /> },
                    { text: "SciPy", icon: <SiScipy className="timeline-tech-icon" /> },
                    { text: "Statsmodels", icon: <SiPython className="timeline-tech-icon" /> },
                    { text: "Pandas", icon: <CiViewTable className="timeline-tech-icon" /> },
                ]
            },
            {
                type: "links",
                content: [
                    { text: "Github", link: interestRatesGithubURL },
                ]
            }
        ]
    }    

    const bayesianCNNData = {
        title: "Bayesian CNN",
        id: "bayesian-cnn",
        oppositeContent: (<>Academic Research <br /> Paper Accepted at IJCNN 2025</>),
    
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
                content: "Paper Accepted at the International Joint Conference on Neural Networks (IJCNN), 2025."
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
            {
                type: "links",
                content: [
                    { text: "Github", link: bayesianCNNGithubURL }
                ]
            }
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
            {
                type: "links",
                content: [
                    { text: "Github", link: boardGameRLGithubURL },
                ]
            }
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
        oppositeContent: (<>RAG Exploration <br /> Applied NLP & Knowledge Graphs</>),

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
            {
                type: "links",
                content: [
                    { text: "Github", link: quantTradingGithubURL },
                ]
            }
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

    const movieRecommenderData = {
        title: "Movie Recommendation System (MLOps)",
        id: "movie-recommender-mlops",
        oppositeContent: (<>Academic Project<br /> Cloud-Native ML Pipeline<br /> Real-Time Scalable Deployment</>),
    
        items: [
            {
                type: "text",
                content: "Designed a production-grade movie recommendation system using an end-to-end MLOps pipeline optimized for cloud infrastructure."
            },
            {
                type: "text",
                content: "Used MLFlow for experiment tracking and model versioning, while provisioning infrastructure using Terraform."
            },
            {
                type: "text",
                content: "Deployed scalable inference services on AWS EC2 using Docker & Kubernetes, with real-time monitoring through Prometheus and Grafana."
            },
            {
                type: "text",
                content: "Implemented distributed hyperparameter tuning with Ray and multi-GPU training via PyTorch, processing over 500K+ daily user interactions."
            },
            {
                type: "text",
                content: "Enabled sub-100ms latency even under 5x traffic spikes and performed inference benchmarking using custom Triton kernels under various quantization and batching configurations."
            },
            {
                type: "collapse",
                title: "Roles / Responsibilities",
                id: "movie-recommender-mlops-roles",
                items: [
                    {
                        type: "list",
                        content: [
                            "Designed and deployed MLOps pipeline with full CI/CD automation.",
                            "Built distributed training setup using PyTorch + Ray on AWS.",
                            "Implemented real-time metrics monitoring with Prometheus-Grafana.",
                            "Benchmarked inference latency with custom Triton backends.",
                            "Handled traffic surges with scalable Dockerized microservices on Kubernetes."
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
            {
                type: "links",
                content: [
                    { text: "Github", link: movieRecommenderGithubURL },
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
                ]
            }
        ]
    }


    const projects = [
        graphDLData,
        bayesianCNNData,
        interestRateModelingData,
        movieRecommenderData,
        rlBoardGamesData,
        electionSentimentData,
        llmBiAssistantData,
        bayesianOptimizationData,
        ragSystemData,
        pinnFnoData,
        llmSummarizationData,
        rnnLstmTradingData,
        binaryTreeData,
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
                    I have worked on a variety of projects, ranging from academic research to personal projects.
                    <br />
                    All of them are in the field of Machine Learning, Data Science and Modelling.
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
