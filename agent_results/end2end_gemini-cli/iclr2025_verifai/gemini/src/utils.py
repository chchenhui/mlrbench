import openai
import os

def get_openai_client():
    """
    Initializes and returns the OpenAI client using the API key from environment variables.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return openai.OpenAI(api_key=api_key)
