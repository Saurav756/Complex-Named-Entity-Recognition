import os
from openai import OpenAI
from retriever import Retriever
from ner_infer import predict_ner

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_prompt(context_chunks, query):
    context = ""
    for chunk in context_chunks:
        ner_tags = predict_ner(chunk)
        tagged_text = " ".join([
            f"<{tag}>{token}</{tag}>" if tag != "O" else token
            for token, tag in ner_tags
        ])
        context += tagged_text + "\n\n"

    return f"""
You are a helpful assistant. The following context contains named entities enclosed in tags.

Context:
{context}

Question: {query}
Answer:
"""

def ask_question(query):
    retriever = Retriever()
    chunks = retriever.retrieve(query, top_k=4)
    prompt = generate_prompt(chunks, query)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    user_question = input("Ask a question about the document: ")
    answer = ask_question(user_question)
    print("\n Answer:\n", answer)
