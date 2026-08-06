from langchain_core.prompts import ChatPromptTemplate

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
"""
You are an expert AI assistant.

You are given the following context.

<context>
{context}
</context>

Generate a concise and accurate summary.

Requirements:
- Maximum 300 words
- Use bullet points where appropriate
- Don't make up information
- Only use the provided context
"""
)

QA_PROMPT = ChatPromptTemplate.from_template(
"""
You are an intelligent assistant.

Answer ONLY using the context below.

<context>
{context}
</context>

Question:
{question}

If the answer is not present in the context, say:

"I couldn't find this information in the provided document."
"""
)