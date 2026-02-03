import google.genai as genai
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def auto_score(context, answer, question):
    prompt = f"""
You are evaluating a Retrieval Augmented Generation (RAG) system.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Return ONLY valid JSON in this exact format:

{{
  "retrieval": 0,
  "grounding": 0,
  "coverage": 0
}}
"""

    response = client.generate_content(
        prompt,
        generation_config={"temperature": 0}
    )

    text = response.text.strip()

    #  Extract JSON safely
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Invalid JSON from model:\n{text}")

    return json.loads(match.group())