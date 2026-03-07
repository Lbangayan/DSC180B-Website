from google import genai
import os
from dotenv import load_dotenv

from pathlib import Path

# Load .env from the correct location
env_path = Path(__file__).parent / "src" / "icl_commerical_model_testing" / ".env"
load_dotenv(env_path)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY not found in environment")
    exit(1)

client = genai.Client(api_key=api_key)

print("Available models:")
print("-" * 80)

try:
    models = client.models.list()
    for model in models:
        print(f"Name: {model.name}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"  Supported methods: {model.supported_generation_methods}")
        print()
except Exception as e:
    print(f"Error listing models: {e}")
