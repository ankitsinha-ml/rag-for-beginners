from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch 

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

#safety check
if not relevant_docs:
    print("No relevant documents found.")
    exit()

#initialising llm (brain)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


#limit context
relevant_docs = relevant_docs[:2]
context = "\n".join([doc.page_content for doc in relevant_docs])

#prompt
prompt = f"""Answer the question using only the context below. If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:"""

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nAnswer:")
print(response)