import re
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader


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

    loader = YoutubeLoader(
        video_id=video_id,
        add_video_info=True,
        language=["en", "en-US", "en-IN", "hi"],
    )

    docs = loader.load()

    if not docs or not docs[0].page_content.strip():
        raise Exception("Transcript not found.")

    return docs


def load_website(url: str):

    loader = WebBaseLoader(url)

    docs = loader.load()

    if not docs:
        raise Exception("Website could not be loaded")

    return docs