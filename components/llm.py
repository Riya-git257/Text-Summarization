
from langchain_groq import ChatGroq

def get_llm(api_key):

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0
    )

    return llm