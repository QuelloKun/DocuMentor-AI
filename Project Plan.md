### Generative AI Chatbot with Fine-Tuning and RAG (Cost-Free)

**Project Title:** `DocuMentor AI` - A Specialized Q&A Chatbot

#### **Phase 0: Foundation & Scoping (Timeline: Week 1)**

**Goal:** Define the project scope, choose a domain with free data, and set up the development environment.

**Steps:**

1. **Choose a Domain:** Select **Option B (Research)**. The **arXiv API** provides free and easy access to hundreds of thousands of research papers. We will focus on a specific, dense sub-field like "Machine Learning" or "Computer Vision." This provides a massive, high-quality dataset at no cost.
    
2. **Define the Core Problem:** State the user story: _"As a student researcher, I want to upload several new research papers and ask specific, technical questions about their methodologies and conclusions to quickly synthesize information for my literature review."_
    
3. **Architect the System (High-Level):** The flow remains the same: User Upload -> Document Processing -> Local Vector Store -> Fine-Tuned LLM -> User Interface.
    
4. **Setup Development Environment:**
    
    - Initialize a **Git** repository on GitHub.
        
    - Set up a Python environment (`venv` or `conda`).
        
    - Install initial core libraries (`jupyter`, `langchain`, `fastapi`, `uvicorn`, `streamlit`).
        
    - Set up free-tier accounts for cloud services where needed (e.g., Hugging Face, Vercel).
        

**Definition of Done:** A clear project scope is documented in a `README.md`, the GitHub repository is created, and the local development environment is functional.

#### **Phase 1: Data Acquisition & Preparation (Timeline: Weeks 2-3)**

**Goal:** Source and structure a high-quality dataset for fine-tuning from arXiv.

**Steps:**

1. **Acquire Raw Data:** Write a **Python** script using the `arxiv` library to download the 500-1000 most relevant papers (PDFs) from your chosen sub-field.
    
2. **Data Extraction:** Use `PyMuPDF` to parse the PDFs and extract clean, structured text.
    
3. **Create a Fine-Tuning Dataset:** Use a powerful, freely accessible model (e.g., via the Hugging Face API or a local Llama 3 instance) to help generate a high-quality Question/Answer dataset from the paper text. Aim for at least 1,000 instruction-following examples.
    
4. **Store the Dataset:** Save the final dataset locally and back it up on your GitHub repository using Git LFS (Large File Storage).
    

**Definition of Done:** A well-structured, high-quality fine-tuning dataset is stored locally and tracked with Git LFS.

#### **Phase 2: Model Fine-Tuning & Evaluation (Timeline: Weeks 4-5)**

**Goal:** Fine-tune an open-source LLM on your local machine or using free cloud credits.

**Steps:**

1. **Select a Base Model:** Choose a high-quality, pre-trained open-source model that can run on consumer hardware. **Llama 3 8B Instruct** or **Mistral 7B Instruct** are excellent choices.
    
2. **Set up Training Environment:**
    
    - **Option A (Local):** If you have a local machine with a capable NVIDIA GPU, set up the environment with CUDA.
        
    - **Option B (Free Cloud):** Use **Google Colab** or Kaggle Kernels, which provide free GPU access for a limited number of hours per week.
        
3. **Write the Fine-Tuning Script:** Use the **Hugging Face Transformers** library, **PyTorch**, and tools like PEFT (Parameter-Efficient Fine-Tuning) to write a script that can efficiently fine-tune the model on your hardware.
    
4. **Launch & Monitor the Training Job:** Run the fine-tuning script locally or in your Colab notebook.
    
5. **Evaluate the Model:** Create an evaluation script to prove that your fine-tuned model provides more accurate, context-aware answers about your research domain than the base model.
    
6. **Store the Fine-Tuned Model:** Save the model adapters (from PEFT) locally and upload them to a **Hugging Face Hub** repository for free model hosting.
    

**Definition of Done:** A fine-tuned model is hosted on Hugging Face Hub, with documented proof of its improved performance.

#### **Phase 3: RAG Pipeline Development (Timeline: Weeks 6-7)**

**Goal:** Build the system that processes user-uploaded documents for real-time Q&A.

**Steps:**

1. **Choose a Vector Database:** Use a free, open-source, and locally run vector database like **ChromaDB** or **FAISS**. These run directly in your Python environment with no external services required.
    
2. **Develop the Ingestion Logic:** Using **LangChain**, write the Python code to process user-uploaded documents, split them into chunks, create embeddings, and store them in your local vector database.
    
3. **Develop the Retrieval Logic:** Write the code to take a user's question, query the vector database for relevant context, and combine it into a prompt for your fine-tuned LLM.
    

**Definition of Done:** A set of Python functions that can successfully process a document and retrieve relevant context for a given question, all running locally.

#### **Phase 4: API & UI Development (Timeline: Weeks 8-9)**

**Goal:** Create a backend service and a user-friendly web interface for your application.

**Steps:**

1. **Design and Implement the API:** Use **FastAPI** to create a simple backend with endpoints for document upload and asking questions.
    
2. **Build the User Interface:** Instead of a complex React frontend, use a simpler, faster tool like **Streamlit** or Gradio. These are Python libraries that allow you to create beautiful data science and AI web apps with just a few lines of Python code.
    
3. **Containerize the Application:** Write a **Dockerfile** that packages your Streamlit UI, FastAPI backend, and all dependencies into a single, portable container.
    

**Definition of Done:** A fully functional, containerized application with a user interface that can be run locally with Docker.

#### **Phase 5: Deployment & Sharing (Timeline: Weeks 10-12)**

**Goal:** Deploy the application to a free cloud platform to create a live, shareable demo.

**Steps:**

1. **Choose a Hosting Platform:** Use **Hugging Face Spaces** or **Vercel's** free tier. Both have excellent support for deploying Docker containers and Streamlit applications at no cost.
    
2. **Build a CI/CD Pipeline:** Use **GitHub Actions** to create a workflow that automatically:
    
    - Runs tests on every push.
        
    - Builds the Docker image.
        
    - Pushes the image to Docker Hub (free for public repositories).
        
    - Deploys the new version of your service to Hugging Face Spaces or Vercel.
        
3. **Deploy the Application:** Run your CI/CD pipeline to make the application live.
    

**Definition of Done:** The application is live and accessible via a public URL, with a CI/CD pipeline for automated updates.

#### **Phase 6: Documentation & Portfolio Presentation (Timeline: Week 13)**

**Goal:** Finalize the project and present it professionally in your portfolio.

**Steps:**

1. **Create a Comprehensive README:** Update your GitHub `README.md` with a high-level overview, a detailed architecture diagram, setup instructions, and a link to the live demo on Hugging Face Spaces/Vercel.
    
2. **Write a Blog Post:** Detail your journey, the challenges of local fine-tuning, and the architectural decisions you made.
    
3. **Prepare for Interviews:** Be ready to talk about every trade-off you made, especially why you chose specific open-source tools and free-tier services to build a production-quality application without a budget.
    

**Definition of Done:** A polished, professional portfolio piece that you can confidently share with any recruiter or hiring manager.