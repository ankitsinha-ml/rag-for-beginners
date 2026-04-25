from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

persist_directory="db/chroma_db"

#load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db= Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

#search for relevant documents
query="What was Nvidia's first graphics accelerator called?"

retriever = db.as_retriever(search_kwargs={"k":3})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold" : 0.3 #only return chunks with cosine similarity >= 0.3
#         }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
#display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


