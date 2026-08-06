from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from components.prompts import QA_PROMPT


def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def build_rag_chain(retriever, llm):

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain