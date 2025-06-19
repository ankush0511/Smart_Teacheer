import os
import logging
import streamlit as st
import wikipediaapi
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from typing import List, Dict, Optional
from dotenv import load_dotenv
from YT_transcipt import process_video, query_vector_db
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EDUCATION_LEVEL = "college"

# Validate environment
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set.")
    st.stop()

# Initialize Groq LLM
try:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.7, max_tokens=800)
except Exception as e:
    st.error(f"Failed to initialize Groq LLM: {e}")
    st.stop()

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia("AcademicExplainer/1.0", "en")

def disambiguate_topic(topic: str) -> str:
    """Add context to ambiguous topics for better search results."""
    topic = topic.strip().lower()
    academic_context = {
        "stack": "stack data structure",
        "agile": "agile software development",
        "benzene": "benzene chemistry",
        "attention mechanism": "attention mechanism neural networks"
    }
    return academic_context.get(topic, topic)

def fetch_wikipedia_explanation(topic: str) -> Optional[str]:
    """Fetch explanation from Wikipedia."""
    try:
        page = wiki.page(topic)
        if page.exists():
            summary = page.summary[:1000]  # Limit to avoid token overflow
            prompt = f"Summarize the following text for a {EDUCATION_LEVEL} student in 150-200 words: {summary}"
            response = llm.invoke(prompt)
            return response.content.strip()
        return None
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed for {topic}: {e}")
        return None

def fetch_duckduckgo_explanation(topic: str) -> str:
    """Fetch explanation from DuckDuckGo as fallback."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{topic} explanation", max_results=1))
            if results:
                content = results[0].get("body", "No content found.")
                prompt = f"Summarize the following text for a {EDUCATION_LEVEL} student in 150-200 words: {content[:1000]}"
                response = llm.invoke(prompt)
                return response.content.strip()
            return f"No reliable content found for {topic}."
    except Exception as e:
        logger.error(f"DuckDuckGo fetch failed for {topic}: {e}")
        return f"Error fetching content for {topic}: {e}"

def fetch_youtube_video(topic: str) -> Dict:
    """Fetch YouTube video link using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.videos(f"{topic} tutorial", max_results=1))
            if results:
                video = results[0]
                return {
                    "url": video.get("content", ""),
                    "title": video.get("title", "Unknown"),
                    "description": video.get("description", "")
                }
            return {}
    except Exception as e:
        logger.error(f"YouTube video fetch failed for {topic}: {e}")
        return {}

def process_syllabus(topics: List[str]) -> List[Dict]:
    """Process syllabus topics for explanations and YouTube links."""
    results = []
    for topic in topics:
        topic = disambiguate_topic(topic)
        # Try Wikipedia first
        explanation = fetch_wikipedia_explanation(topic)
        if not explanation:
            # Fallback to DuckDuckGo
            explanation = fetch_duckduckgo_explanation(topic)
        
        # Fetch YouTube video
        video_data = fetch_youtube_video(topic)
        
        results.append({
            "topic": topic,
            "explanation": explanation,
            "video_url": video_data.get("url", "No video found"),
            "video_title": video_data.get("title", "Unknown")
        })
        
        # Rate limiting
        time.sleep(2)
    
    return results

def process_youtube_video(video_url: str, title: str = "Unknown") -> Dict:
    """Process a YouTube video for transcript and summary."""
    try:
        result = process_video(video_url, title)
        return result
    except Exception as e:
        logger.error(f"Error processing YouTube video {video_url}: {e}")
        return {
            "video_url": video_url,
            "transcript": f"Error generating transcript: {e}",
            "summary": f"Error generating summary: {e}",
            "stored": False
        }

# Streamlit app
st.title("Academic Explainer")

# Sidebar for option selection
option = st.sidebar.selectbox(
    "Choose an option:",
    ["Topic Description & YouTube Link", "YouTube Transcript & Summary"]
)

if option == "Topic Description & YouTube Link":
    st.header("Topic Description & YouTube Link")
    syllabus_input = st.text_input("Enter syllabus topics (comma-separated, e.g., Photosynthesis, Quick Sort):")
    
    if st.button("Process Topics"):
        if syllabus_input:
            topics = [topic.strip() for topic in syllabus_input.split(",")]
            with st.spinner("Processing topics..."):
                results = process_syllabus(topics)
                for result in results:
                    st.subheader(f"Topic: {result['topic'].capitalize()}")
                    st.write("**Explanation:**")
                    st.write(result['explanation'])
                    st.write("**YouTube Link:**")
                    st.write(result['video_url'])
                    st.write("**Video Title:**")
                    st.write(result['video_title'])
                    st.markdown("---")
        else:
            st.warning("Please enter at least one topic.")

elif option == "YouTube Transcript & Summary":
    st.header("YouTube Transcript & Summary")
    video_url = st.text_input("Enter a YouTube video URL:")
    title = st.text_input("Enter a title for the video (optional):", value="Unknown")
    
    if st.button("Process Video"):
        if video_url:
            with st.spinner("Processing video..."):
                result = process_youtube_video(video_url, title)
                st.subheader("Video Details")
                st.write("**Video URL:**")
                st.write(result['video_url'])
                st.write("**Transcript (excerpt):**")
                transcript = result['transcript']
                st.write(transcript[:500] + '...' if len(transcript) > 500 else transcript)
                st.write("**Summary:**")
                st.write(result['summary'])
                st.write(f"**Stored in vector DB:** {'✅' if result['stored'] else '❌'}")
                
                # Query input
                st.subheader("Ask a Question About the Video")
                query = st.text_input("Enter your question (or leave blank to skip):")
                if query and st.button("Submit Query"):
                    with st.spinner("Searching database..."):
                        responses = query_vector_db(query)
                        if responses:
                            st.subheader("Query Results")
                            for resp in responses:
                                st.write("**Video URL:**")
                                st.write(resp['video_url'])
                                st.write("**Title:**")
                                st.write(resp['title'])
                                st.write("**Transcript (excerpt):**")
                                st.write(resp['transcript'][:500] + '...' if len(resp['transcript']) > 500 else resp['transcript'])
                                st.write("**Summary:**")
                                st.write(resp['summary'])
                                st.write(f"**Relevance Score:** {resp['score']:.2f}")
                                st.markdown("---")
                        else:
                            st.info("No relevant information found in the database.")
        else:
            st.warning("Please enter a valid YouTube URL.")

st.sidebar.markdown("---")
st.sidebar.info("Built with Groq, Wikipedia, DuckDuckGo, and ChromaDB.")