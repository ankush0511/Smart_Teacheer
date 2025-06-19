import os
import re
import logging
import time
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import uuid
from typing import Dict, Optional, List
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EDUCATION_LEVEL = "college"
COLLECTION_NAME = "youtube_videos"

# Validate environment
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Initialize Groq LLM
try:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.7,
        max_tokens=200
    )
except Exception as e:
    logger.error(f"Failed to initialize Groq LLM: {e}")
    raise

# Initialize ChromaDB and embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
            query = parse_qs(parsed_url.query)
            return query.get("v", [None])[0]
        elif parsed_url.hostname in ["youtu.be"]:
            return parsed_url.path.lstrip("/")
        logger.warning(f"Invalid YouTube URL: {url}")
        return None
    except Exception as e:
        logger.error(f"Error extracting video ID from {url}: {e}")
        return None

def get_video_transcript(video_id: str) -> Optional[str]:
    """Fetch transcript for a YouTube video with retry."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        logger.warning(f"Transcript unavailable for video {video_id}: {e}")
        return None

def scrape_video_description(url: str) -> Optional[str]:
    """Scrape video description using requests and BeautifulSoup."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find description in YouTube's metadata
        meta_description = soup.find("meta", {"name": "description"})
        if meta_description and meta_description.get("content"):
            return meta_description["content"]
        
        # Fallback to description in page content
        description_div = soup.find("div", {"id": "description"})
        if description_div:
            return description_div.get_text(strip=True)
        
        return "No description available."
    except Exception as e:
        logger.error(f"Failed to scrape description for {url}: {e}")
        return None

def summarize_content(content: str, video_url: str) -> str:
    """Summarize transcript or description using Groq LLM."""
    if not content or content in ["No transcript available.", "No description available."]:
        return f"No content available for summarization of video {video_url}."
    
    prompt = f"""
    Summarize the following content in 150-200 words. Focus on key concepts, examples, and explanations for a {EDUCATION_LEVEL} student. Keep it concise and clear.

    Content: {content[:4000]}  # Truncate to avoid token limits
    """
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Failed to summarize content: {e}")
        return f"Failed to summarize video due to error: {e}"

def store_in_vector_db(
    video_url: str, transcript: str, summary: str, metadata: Dict
) -> None:
    """Store video data in ChromaDB."""
    try:
        # doc_id = str(uuid.uuid4())
        document = Document(
            page_content=f"Video URL: {video_url}\nTranscript: {transcript}\nSummary: {summary}",
            metadata=metadata,
            # id=doc_id
        )
        vector_store.add_documents([document])
        logger.info(f"Stored data for video {video_url} in vector database.")
    except Exception as e:
        logger.error(f"Failed to store data in vector database: {e}")
        raise

def query_vector_db(query: str, top_k: int = 1) -> List[Dict]:
    """Retrieve relevant information from the vector database."""
    try:
        results = vector_store.similarity_search_with_score(query, k=top_k)
        responses = []
        for doc, score in results:
            responses.append({
                "video_url": doc.metadata.get("video_url", "N/A"),
                "title": doc.metadata.get("title", "Unknown"),
                "transcript": doc.page_content.split("Transcript: ")[1].split("\nSummary: ")[0],
                "summary": doc.page_content.split("Summary: ")[1],
                "score": score
            })
        return responses
    except Exception as e:
        logger.error(f"Vector database query failed: {e}")
        return []

def process_video(video_url: str, title: str = "Unknown") -> Dict:
    """Process a YouTube video: generate transcript, summarize, and store."""
    try:
        # Extract video ID
        video_id = extract_video_id(video_url)
        if not video_id:
            return {
                "video_url": video_url,
                "transcript": "Invalid YouTube URL",
                "summary": "No summary generated due to invalid URL",
                "stored": False
            }
        
        # Try fetching transcript
        transcript = get_video_transcript(video_id)
        content = transcript
        
        # Fallback to description if transcript is unavailable
        if not transcript:
            logger.info(f"Falling back to video description for {video_url}")
            content = scrape_video_description(video_url) or "No description available."
        
        # Summarize content
        summary = summarize_content(content, video_url)
        
        # Prepare metadata
        metadata = {
            "video_url": video_url,
            "title": title,
            "education_level": EDUCATION_LEVEL
        }
        
        # Store in vector database
        store_in_vector_db(video_url, content, summary, metadata)
        
        return {
            "video_url": video_url,
            "transcript": content,
            "summary": summary,
            "stored": True
        }
    except Exception as e:
        logger.error(f"Error processing video {video_url}: {e}")
        return {
            "video_url": video_url,
            "transcript": f"Error generating transcript: {e}",
            "summary": f"Error generating summary: {e}",
            "stored": False
        }

