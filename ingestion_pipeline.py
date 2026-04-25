import os
from dotenv import load_dotenv

# document loaders
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# text splitter
from langchain_text_splitters import CharacterTextSplitter

#embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings 

# vector store (Chroma)
from langchain_chroma import Chroma

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    #checks if directory exits or not
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exits. Please create it and add your company files.")

    #load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")


    for i,doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f" Source: {doc.metadata.get('source','N/A')}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview:{doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")

    return documents    


def main():
    print("Main Function")
    

    #1. Loading the files
    documents = load_documents(docs_path="docs")
if __name__ == "__main__":
    main()    