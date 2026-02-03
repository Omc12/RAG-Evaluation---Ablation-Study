import google.genai as genai
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

def generate_answer(context, question):
    prompt = f"""
You are a research assistant.
Answer ONLY using the context.
If the answer is not present, say "Not enough information".

CONTEXT:
{context}

QUESTION:
{question}

Answer concisely.
"""
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0,
            "max_output_tokens": 500
        }
    )
    return response.text
