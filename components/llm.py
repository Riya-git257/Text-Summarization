
from langchain_groq import ChatGroq

def get_llm(api_key):

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=api_key,
        temperature=0
    )

    return llm