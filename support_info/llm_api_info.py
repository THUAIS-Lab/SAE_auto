import os

api_key_file = os.getenv("LLM_API_KEY_FILE")
base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
model_name = os.getenv("LLM_MODEL", "gpt-4.1-mini")
