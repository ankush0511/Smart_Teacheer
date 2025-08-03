from urllib.parse import urlparse, parse_qs
from typing import Optional, Tuple
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import time
import logging
from dataclasses import dataclass
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()


logger = logging.getLogger(__name__)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@dataclass
class FetchedTranscriptSnippet:
    text: str
    start: float
    duration: float

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

def get_transcript_and_summary(video_id: str) -> Tuple[str, str]:
    """Fetch transcript and generate summary for a YouTube video."""
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.list(video_id)
    final_trans = ""
    final_sum = ""

    for transcript in transcript_list:
        lan = transcript.language_code
        res = transcript.fetch()
        snippets = [FetchedTranscriptSnippet(text=item.text, start=item.start, duration=item.duration) for item in res]
        combined_text = " ".join(snippet.text for snippet in snippets)

        if lan == 'hi':
            text_split = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=200)
            tsplit = text_split.split_text(combined_text)
            for i in tsplit:
                llm = ChatGroq(model='gemma2-9b-it', api_key=GROQ_API_KEY)
                prompt = f"""Translate the following Hindi text into fluent English. Return only the translated English text without any explanation:\n{i}"""
                result = llm.invoke(prompt)
                time.sleep(1)
                final_trans += result.content + " "
                prompt = f"""Summarize the following text in 1-2 lines:\n{i}"""
                result = llm.invoke(prompt)
                final_sum += result.content + "\n"
                time.sleep(3)
        else:
            final_trans = combined_text
            text_split = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=200)
            tsplit = text_split.split_text(combined_text)
            for i in tsplit:
                llm = ChatGroq(model='gemma2-9b-it', api_key=GROQ_API_KEY)
                prompt = f"""Summarize the following text in 1-2 lines:\n{i}"""
                result = llm.invoke(prompt)
                final_sum += result.content + "\n"
    return final_trans, final_sum