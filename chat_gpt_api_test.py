import os
from dotenv import load_dotenv
from helpers import ChatGPT

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Example usage
if __name__ == "__main__":
    gpt = ChatGPT(api_key = api_key)
    response = gpt.chat_with_gpt("Hello GPT. Tell me a bit about your history and how you have been created.")
    print(f"ChatGPT: {response}")
