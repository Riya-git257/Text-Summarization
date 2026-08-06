import re
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url: str):
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def load_youtube(url: str):

    video_id = extract_video_id(url)

    if not video_id:
        raise Exception("Invalid YouTube URL")

    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=["en", "hi"]
        )

    except Exception as e:
        raise Exception(f"Could not fetch transcript: {e}")

    text = " ".join(chunk["text"] for chunk in transcript)

    return [
        Document(
            page_content=text,
            metadata={
                "source": url,
                "video_id": video_id
            }
        )
    ]


def load_website(url: str):

    loader = WebBaseLoader(url)

    docs = loader.load()

    if not docs:
        raise Exception("Website could not be loaded")

    return docs