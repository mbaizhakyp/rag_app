import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub


st.set_page_config(page_title="RAG Web Demo", page_icon="", layout="wide")
st.title("RAG Question-Answering Demo")

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is not set in the environment variables")
    st.stop()

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

url = st.text_input("Enter a URL to load documents from:",
                    value="https://www.govinfo.gov/content/pkg/CDOC-110hdoc50/html/CDOC-110hdoc50.htm")

if st.button("Initialize RAG Button"):
    with st.spinner("Loading and processing documents..."):
        loader = WebBaseLoader(url)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
        st.session_state.vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
        st.success("RAG system initialized successfully")

if st.session_state.vectorstore is not None:
    llm = ChatOpenAI(model = "gpt-4o")

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a helpful assistant that can answer questions about the US Constitution. Use the provided context to answer the question. IMPORTANT: If the context does not provide the answer, say 'I don't know' and don't make up the answer."),
    #     ("user", "Question: {question}\nContext: {context}")
    # ])
    prompt = hub.pull("rlm/rag-prompt")


    chain = prompt | llm

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ask a Question")
        question = st.text_area("Enter your question:")

        if st.button("Get Answer"):
            if question:
                with st.spinner("Generating Answer..."):
                    retriever = st.session_state.vectorstore.as_retriever()
                    docs = retriever.invoke(question)
                    docs_content = "\n\n".join(doc.page_content for doc in docs)

                    response = chain.invoke({
                        "question": question,
                        "context": docs_content
                    })

                    st.session_state.last_response = response.content
                    st.session_state.last_context = docs
            else:
                st.warning("Please enter a question:")
        
    with col2:
        st.subheader("Answer")
        if 'last_response' in st.session_state:
            st.write(st.session_state.last_response)

            with st.expander("Show Retrieved Content"):
                for i, doc in enumerate(st.session_state.last_context, 1):
                    st.markdown(f"**Relevant Document {i}:**")
                    st.markdown(doc.page_content)
                    st.markdown("---")