import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

##streamlit app
st.set_page_config(page_title="Langchain: text summarize from Url and Youtube")
st.title("Langchain: text summarize from Url and Youtube")
st.subheader("Summarize URL")
st.write("Please enter Groq Api key")

#Get api key and url field 

with st.sidebar:
    groq_api_key=st.text_input("GROQ API KEY",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

llm=ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

template="""
Provide a summary of the following content in 300 words
Content:{text}

"""

prompt=PromptTemplate(template=template,input_variables=["text"])

if st.button("Summarize the content from YT or website"):

    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url")
    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False)

                text_splitter=RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=200)
                docs=loader.load()
                split=text_splitter.split_documents(docs)

                ## Chain
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                summary=chain.run(split)

                st.success(summary)
        except Exception as e:
            st.exception(f"Exception:{e}")

    
