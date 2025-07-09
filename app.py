# Code Review Buddy - Complete Implementation
# A RAG-powered AI agent that analyzes GitHub repos and provides code review suggestions

import os
import streamlit as st
import requests
import base64
from typing import List, Dict, Any
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import tempfile
import zipfile
import io

# =============================================================================
# STEP 1: Setup and Configuration
# =============================================================================

# Environment variables you'll need:
# OPENAI_API_KEY
# GITHUB_TOKEN (optional, for private repos)

class CodeReviewBuddy:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.github_token = os.getenv("GITHUB_TOKEN")
        
        if not self.openai_api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        openai.api_key = self.openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Code review prompts for different aspects
        self.review_prompts = {
            "security": """
            Analyze this code for security vulnerabilities. Look for:
            - SQL injection risks
            - XSS vulnerabilities
            - Authentication/authorization issues
            - Input validation problems
            - Hardcoded secrets
            - Insecure dependencies
            
            Code context: {context}
            
            Provide specific, actionable security recommendations.
            """,
            
            "performance": """
            Analyze this code for performance issues. Look for:
            - Inefficient algorithms
            - Memory leaks
            - Unnecessary loops or operations
            - Database query optimization
            - Caching opportunities
            - Resource management
            
            Code context: {context}
            
            Provide specific performance improvement suggestions.
            """,
            
            "style": """
            Analyze this code for style and best practices. Look for:
            - Code organization and structure
            - Naming conventions
            - Documentation quality
            - Error handling
            - Code duplication
            - Design patterns usage
            
            Code context: {context}
            
            Provide specific style and best practice recommendations.
            """
        }

# =============================================================================
# STEP 2: GitHub Repository Processing
# =============================================================================

    def fetch_github_repo(self, repo_url: str) -> Dict[str, Any]:
        """
        Fetch repository contents from GitHub API
        """
        # Extract owner and repo name from URL
        parts = repo_url.replace("https://github.com/", "").split("/")
        if len(parts) < 2:
            raise ValueError("Invalid GitHub URL format")
        
        owner, repo = parts[0], parts[1]
        
        # GitHub API endpoint
        api_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
        
        headers = {}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        response = requests.get(api_url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch repository: {response.status_code}")
        
        return {
            "owner": owner,
            "repo": repo,
            "zip_content": response.content
        }

    def extract_code_files(self, zip_content: bytes) -> List[Dict[str, str]]:
        """
        Extract and filter code files from repository zip
        """
        code_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
            '.sql', '.html', '.css', '.vue', '.svelte'
        }
        
        code_files = []
        
        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.is_dir():
                    continue
                
                file_path = file_info.filename
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext in code_extensions:
                    try:
                        with zip_ref.open(file_info) as file:
                            content = file.read().decode('utf-8', errors='ignore')
                            
                            # Skip very large files (>50KB)
                            if len(content) > 50000:
                                continue
                                
                            code_files.append({
                                "path": file_path,
                                "content": content,
                                "extension": file_ext
                            })
                    except Exception as e:
                        # Skip files that can't be read
                        continue
        
        return code_files

# =============================================================================
# STEP 3: RAG System Implementation
# =============================================================================

    def create_vector_store(self, code_files: List[Dict[str, str]]) -> FAISS:
        """
        Create vector store from code files for RAG
        """
        documents = []
        
        for file_info in code_files:
            # Create chunks with file context
            chunks = self.text_splitter.split_text(file_info["content"])
            
            for chunk in chunks:
                # Add file context to each chunk
                doc_content = f"File: {file_info['path']}\n\n{chunk}"
                documents.append(doc_content)
        
        if not documents:
            raise ValueError("No code files found to analyze")
        
        # Create vector store
        vectorstore = FAISS.from_texts(
            documents, 
            self.embeddings,
            metadatas=[{"source": f"chunk_{i}"} for i in range(len(documents))]
        )
        
        return vectorstore

    def analyze_code_aspect(self, vectorstore: FAISS, aspect: str, query: str) -> str:
        """
        Analyze specific aspect of code using RAG
        """
        # Create retrieval chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Get relevant code chunks
        relevant_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate analysis using OpenAI
        prompt = self.review_prompts[aspect].format(context=context)
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert code reviewer with deep knowledge of security, performance, and best practices."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return response.choices[0].message.content

# =============================================================================
# STEP 4: Main Analysis Pipeline
# =============================================================================

    def analyze_repository(self, repo_url: str) -> Dict[str, Any]:
        """
        Complete repository analysis pipeline
        """
        try:
            # Step 1: Fetch repository
            st.write("📥 Fetching repository...")
            repo_data = self.fetch_github_repo(repo_url)
            
            # Step 2: Extract code files
            st.write("📂 Extracting code files...")
            code_files = self.extract_code_files(repo_data["zip_content"])
            
            if not code_files:
                return {"error": "No code files found in repository"}
            
            st.write(f"Found {len(code_files)} code files")
            
            # Step 3: Create vector store
            st.write("🔍 Creating knowledge base...")
            vectorstore = self.create_vector_store(code_files)
            
            # Step 4: Analyze different aspects
            results = {
                "repo_info": {
                    "owner": repo_data["owner"],
                    "repo": repo_data["repo"],
                    "files_analyzed": len(code_files)
                },
                "analyses": {}
            }
            
            aspects = ["security", "performance", "style"]
            queries = {
                "security": "security vulnerabilities authentication authorization input validation",
                "performance": "performance optimization algorithms memory database queries",
                "style": "code style best practices naming conventions documentation"
            }
            
            for aspect in aspects:
                st.write(f"🔍 Analyzing {aspect}...")
                try:
                    analysis = self.analyze_code_aspect(vectorstore, aspect, queries[aspect])
                    results["analyses"][aspect] = analysis
                except Exception as e:
                    results["analyses"][aspect] = f"Error analyzing {aspect}: {str(e)}"
            
            return results
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

# =============================================================================
# STEP 5: Streamlit Web Interface
# =============================================================================

def main():
    st.set_page_config(
        page_title="Code Review Buddy",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Code Review Buddy")
    st.markdown("**AI-powered code review for GitHub repositories**")
    
    # Initialize the code review buddy
    try:
        buddy = CodeReviewBuddy()
    except ValueError as e:
        st.error(f"Configuration error: {e}")
        st.info("Please set your OPENAI_API_KEY environment variable")
        return
    
    # Input section
    st.header("📝 Repository Analysis")
    
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/username/repository",
        help="Enter a public GitHub repository URL"
    )
    
    if st.button("🚀 Analyze Repository", type="primary"):
        if not repo_url:
            st.error("Please enter a repository URL")
            return
        
        if not repo_url.startswith("https://github.com/"):
            st.error("Please enter a valid GitHub repository URL")
            return
        
        # Perform analysis
        with st.spinner("Analyzing repository..."):
            results = buddy.analyze_repository(repo_url)
        
        # Display results
        if "error" in results:
            st.error(results["error"])
        else:
            display_results(results)

def display_results(results: Dict[str, Any]):
    """
    Display analysis results in a nice format
    """
    st.success("✅ Analysis complete!")
    
    # Repository info
    repo_info = results["repo_info"]
    st.subheader(f"📊 {repo_info['owner']}/{repo_info['repo']}")
    st.info(f"Analyzed {repo_info['files_analyzed']} code files")
    
    # Analysis results
    analyses = results["analyses"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔒 Security Analysis")
        st.write(analyses.get("security", "No analysis available"))
    
    with col2:
        st.subheader("⚡ Performance Analysis")
        st.write(analyses.get("performance", "No analysis available"))
    
    with col3:
        st.subheader("✨ Style Analysis")
        st.write(analyses.get("style", "No analysis available"))

# =============================================================================
# STEP 6: Requirements and Setup Instructions
# =============================================================================


if __name__ == "__main__":
    main()
