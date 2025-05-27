import cohere
import os
import numpy as np
import httpx
import together
from together import Together
from .config import COHERE_API_KEY
from .config import TOGATHERAI_API_KEY

co = cohere.ClientV2(COHERE_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CV_PATH = os.path.join(BASE_DIR, "..", "cv.txt")

DOCUMENT_CONTENT = None


def store_user_document(text: str):
    global DOCUMENT_CONTENT
    DOCUMENT_CONTENT = text
    return True

# Load and chunk the CV
def load_uploaded_chunks(chunk_size=3):
    global DOCUMENT_CONTENT
    if not DOCUMENT_CONTENT:
        raise ValueError("No document uploaded yet.")

    lines = [line.strip() for line in DOCUMENT_CONTENT.splitlines() if line.strip()]
    
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = " ".join(lines[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

# Embed each chunk using Cohere
def embed_chunks(chunks):
    response = co.embed(texts=chunks, model="embed-english-v3.0", input_type='search_document', embedding_types= ['float'])

    embeddings = response.embeddings.float_
    return list(zip(chunks, embeddings))

# Calculate cosine similarity
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Answer a question using RAG
async def generate_answer_from_transcription(question: str):
    chunks = load_uploaded_chunks()
    embedded_chunks = embed_chunks(chunks)

    # Embed the question
    response = co.embed(texts=[question], model="embed-english-v3.0", input_type='search_document', embedding_types= ['float'])
    q_embed = response.embeddings.float_[0]

    # Find top-3 similar chunks
    scored = [
        (chunk, cosine_similarity(q_embed, emb))
        for chunk, emb in embedded_chunks
    ]
    top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:3]
    context = "\n".join([chunk for chunk, _ in top_chunks])

    # Compose prompt
    prompt = f"""Answer the following question using the CV content below.

            CV:
            {context}

            Question: {question}
            Answer:"""

    answer = await call_together_llm(prompt)
    
    # For now, mock the answer — we’ll plug a model in next
    return {answer}




async def call_together_llm(prompt: str) -> str:
    client = Together(api_key = TOGATHERAI_API_KEY)

    response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    messages=[
      {"role": "system", "content": 
       """You are an assistant that answers questions about a CV.
      Make your answer as positive as possible to help him get a job. Your response should be seamless with no bullet points, numbering or any other similar things. 
      Keep your answers as short as possible and on point"""
      },
      {"role": "user", "content": f"{prompt}"},
        ],
    )
    return response.choices[0].message.content




async def answer_question(query: str) -> str:
    # TODO: Embed query → retrieve chunks → generate answer
    return f"This is a placeholder answer for the question: '{query}'"