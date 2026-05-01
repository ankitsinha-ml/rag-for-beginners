from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_answer(query, context, history):
    prompt = f"""You are a helpful assistant. Answer using only the context below.
If the answer is not in the context, say "I don't know".

Previous conversation:
{history}

Context:
{context}

Question: {query}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=256)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)