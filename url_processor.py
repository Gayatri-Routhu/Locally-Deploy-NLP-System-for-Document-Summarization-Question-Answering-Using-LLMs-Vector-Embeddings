import os
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain
from langchain import PromptTemplate
import streamlit as st

class URLProcessor:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama3.2:3b",
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        """
        Initializes the URLProcessor with embedding models, LLM, and vector store.

        Args:
            model_name (str): HuggingFace model name for embeddings.
            device (str): Device to run the model on ('cpu' or 'cuda').
            encode_kwargs (dict): Additional keyword arguments for encoding.
            llm_model (str): Local LLM model name for ChatOllama.
            llm_temperature (float): Temperature setting for the LLM.
            qdrant_url (str): URL for the Qdrant instance.
            collection_name (str): Name of the Qdrant collection.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Initialize Embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        # Initialize Local LLM
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.llm_temperature,
        )

        # Initialize Qdrant client and vector store
        self.db = Qdrant(
            client=None,
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            url=self.qdrant_url,
            prefer_grpc=False,
        )

        # Define summarization prompt
        self.summary_prompt = PromptTemplate(
            template="""Summarize the following article:

{article}

Summary:
""",
            input_variables=["article"],
        )
        self.summary_chain = LLMChain(llm=self.llm, prompt=self.summary_prompt)

    def fetch_and_summarize(self, url: str) -> str:
        """
        Fetches article content from the URL and summarizes it.

        Args:
            url (str): URL of the article.
        Returns:
            str: Summary of the article.
        """
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            if not docs:
                raise ValueError("Failed to fetch content from URL.")

            article_content = docs[0].page_content
            summary = self.summary_chain.run({"article": article_content})
            return summary
        except Exception as e:
            raise ValueError(f"Error fetching or summarizing the article: {e}")

    def create_embeddings_from_url(self, url: str) -> str:
        """
        Creates embeddings from the content fetched from a URL and stores them in Qdrant.

        Args:
            url (str): URL of the article.
        Returns:
            str: Success message upon completion.
        """
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=250
            )
            splits = text_splitter.split_documents(docs)

            # Create and store embeddings in Qdrant
            Qdrant.from_documents(
                splits,
                self.embeddings,
                url=self.qdrant_url,
                prefer_grpc=False,
                collection_name=self.collection_name,
            )
            return "✅ URL content successfully stored in Qdrant!"
        except Exception as e:
            raise ConnectionError(f"Error processing URL and creating embeddings: {e}")

    def get_response_from_url(self, query: str) -> str:
        """
        Retrieves answers using RAG from content stored from URL.

        Args:
            query (str): User query.
        Returns:
            str: Chatbot response.
        """
        try:
            retriever = self.db.as_retriever(search_kwargs={"k": 1})
            response = retriever.run(query)
            return response
        except Exception as e:
            raise ValueError(f"Error while generating response: {e}")
