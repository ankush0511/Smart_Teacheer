from typing import Optional, Dict
import logging
from duckduckgo_search import DDGS
import wikipediaapi
from logger import logger
from llm import llm


wiki = wikipediaapi.Wikipedia("AcademicExplainer/1.0", "en")

EDUCATION_LEVEL = "college"

def fetch_wikipedia_explanation(topic: str) -> Optional[str]:
    """Fetch explanation from Wikipedia."""
    try:
        page = wiki.page(topic)
        if page.exists():
            summary = page.summary[:1500]  # Increased limit for more context
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
            results = list(ddgs.text(f"{topic} explanation", max_results=2))  # Increased results for reliability
            if results:
                content = " ".join([result.get("body", "") for result in results])
                prompt = f"Summarize the following text for a {EDUCATION_LEVEL} student in 150-200 words: {content[:1500]}"
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