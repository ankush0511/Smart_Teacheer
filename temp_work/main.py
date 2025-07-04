import os
import logging
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
    raise ValueError("GROQ_API_KEY is not set.")

# Initialize Groq LLM
llm = ChatGroq(model_name="gemma2-9b-it", api_key=GROQ_API_KEY)

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia("AcademicExplainer/1.0", "en")

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
        # Try Wikipedia first
        explanation = fetch_wikipedia_explanation(topic)
        if not explanation:
            # Fallback to DuckDuckGo
            explanation = fetch_duckduckgo_explanation(topic)
            if not explanation:
                explanation = llm.invoke(topic)
        
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

def display_menu():
    """Display user choice menu."""
    print("\nChoose an option:")
    print("1. Get description of topics and YouTube links")
    print("2. Get YouTube transcript and summary")
    print("3. Exit")

def main():
    """Main function to handle user choices."""
    while True:
        display_menu()
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if choice == "3":
            print("Exiting program.")
            break
        
        if choice == "1":
            # Process syllabus for explanations and YouTube links
            syllabus = input("Enter syllabus topics (comma-separated, e.g., Photosynthesis,Quick Sort): ").strip().split(",")
            results = process_syllabus([topic.strip() for topic in syllabus])
            for result in results:
                print(f"""
                📚 Topic: {result['topic']}
                🧠 Explanation: {result['explanation']}
                ▶️ YouTube Link: {result['video_url']}
                🎥 Video Title: {result['video_title']}
                """)
        
        elif choice == "2":
            # Process YouTube video for transcript and summary
            video_url = input("Enter a YouTube video URL: ").strip()
            title = input("Enter a title for the video (optional, press Enter to skip): ").strip() or "Unknown"
            result = process_youtube_video(video_url, title)
            print(f"""
            ▶️ Video URL: {result['video_url']}
            📜 Transcript:
            {result['transcript'][:500] + '...' if len(result['transcript']) > 500 else result['transcript']}
            📝 Summary: {result['summary']}
            💾 Stored in vector DB: {'✅' if result['stored'] else '❌'}
            """)
            
            # Query the database
            query = input("Ask a question about the video (or 'skip' to continue): ").strip()
            if query.lower() != "skip":
                responses = query_vector_db(query)
                if responses:
                    print("\n🔍 Query Results:")
                    for resp in responses:
                        print(f"""
                        Video URL: {resp['video_url']}
                        Title: {resp['title']}
                        Transcript (excerpt): {resp['transcript'][:500] + '...' if len(resp['transcript']) > 500 else resp['transcript']}
                        Summary: {resp['summary']}
                        Relevance Score: {resp['score']:.2f}
                        """)
                else:
                    print("No relevant information found in the database.")
            
            # Rate limiting
            time.sleep(2)
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()