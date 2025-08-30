from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from typing import Optional
from langchain_core.documents import Document
import chromadb
import os
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class KBase:

    def __init__(
        self,
        source_directory: str,
        destination_directory: str = "./embedded_data",
        collection_name: str = "default_knowledge_base",
        embedding_model: Optional[Embeddings] = None,
    ):
        if not os.path.exists(source_directory):
            raise ValueError(f"Source directory {source_directory} does not exist.")

        self.source_directory = os.path.abspath(source_directory)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.destination_directory = os.path.abspath(destination_directory)

    def _load_data(self, omit_existing: bool = False):
        if os.path.isdir(self.source_directory):
            log.debug(f"Loading documents from directory: {self.source_directory}")
            loader = DirectoryLoader(
                self.source_directory, glob="**/*.pdf", loader_cls=PyPDFLoader
            )

        elif os.path.isfile(self.source_directory) and self.source_directory.endswith(
            ".pdf"
        ):
            log.debug(f"Loading document from PDF file: {self.source_directory}")
            loader = PyPDFLoader(self.source_directory)

        documents = loader.load()
        log.debug(f"Loaded {len(documents)} documents from source.")

        return documents

    def _remove_existing_docs(self, documents: list[Document]):
        vector_store = self.get_vector_store()
        if vector_store is None:
            log.debug("Vector store does not exist. All files are new.")
            return documents
        requested_file_paths = set([doc.metadata.get("source") for doc in documents])

        results = vector_store.get(
            where={"source": {"$in": list(requested_file_paths)}}
        )

        existing_file_paths = set([meta["source"] for meta in results["metadatas"]])

        diff = requested_file_paths - existing_file_paths
        log.debug(f"Existing files: {existing_file_paths}")
        log.debug(f"Requested files: {requested_file_paths}")
        log.debug(f"Files to be added: {diff}")

        documents = [doc for doc in documents if doc.metadata.get("source") in diff]
        log.debug(f"After omitting, {len(documents)} documents remain to be added.")

        return documents

    def _split_data(
        self, documents: list[Document], chunk_size=1000, chunk_overlap=100
    ):
        log.debug(
            f"Splitting documents into chunks (size: {chunk_size}, overlap: {chunk_overlap})"
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs = text_splitter.split_documents(documents)
        return docs

    def _create_vector_store(self, documents: list[Document]):
        persist_directory = self.destination_directory
        log.debug(f"Creating vector store at: {persist_directory}")
        if not os.path.exists(persist_directory):
            log.warning(
                f"Vector store does not exist. Creating directory: {persist_directory}"
            )
            os.makedirs(persist_directory)

        vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embedding_model,
        )

        if len(documents) == 0:
            log.info("No new documents to add to the vector store.")
            return vector_store

        log.debug(f"Adding {len(documents)} documents to the vector store.")
        vector_store.add_documents(documents)
        return vector_store

    def embed_and_store(self):
        if self.embedding_model is None:
            log.error("Embedding model is not provided.")
            raise ValueError("Embedding model must be provided.")
        documents = self._load_data(omit_existing=True)
        documents = self._remove_existing_docs(documents)
        chunks = self._split_data(documents)
        vector_store = self._create_vector_store(chunks)
        return vector_store

    def get_vector_store(self, collection_name: str = None):
        persist_directory = self.destination_directory
        log.debug(f"Accessing vector store at: {persist_directory}")
        if not os.path.exists(persist_directory):
            log.error(f"Vector store directory does not exist: {persist_directory}")
            return None

        vector_store = Chroma(
            collection_name=(
                self.collection_name if collection_name is None else collection_name
            ),
            persist_directory=persist_directory,
            embedding_function=self.embedding_model,
        )
        return vector_store

    def list_collections(self):
        persist_directory = self.destination_directory
        if not os.path.exists(persist_directory):
            raise ValueError(
                f"Vector store directory {persist_directory} does not exist."
            )

        client = chromadb.PersistentClient(path=persist_directory)
        collections = client.list_collections()
        return collections

    def get_retriever(self, vector_store, k=5):
        log.debug(f"Creating retriever with top {k} results.")
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        return retriever
