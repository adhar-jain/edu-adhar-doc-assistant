import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")

print("KEY:", key)

genai.configure(api_key=key)

for m in genai.list_models():
    print(m.name)
