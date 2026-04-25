import os
from dotenv import load_dotenv

# document loaders
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# text splitter
from langchain_text_splitters import CharacterTextSplitter

#embeddings
from  langchain.embeddings import HuggingFaceEmbeddings

# vector store (Chroma)
from langchain_chroma import Chroma

load_dotenv()

def main():
    print("Main Function")

if __name__ == "__main__":
    main()    