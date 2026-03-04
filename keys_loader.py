import os
from dotenv import load_dotenv

# This looks for the .env file in your folder
load_dotenv()

def get_secret(key_name):
    value = os.getenv(key_name)
    if not value:
        print(f"Warning: {key_name} not found in .env file!")
    return value

# Example usage:
# password = get_secret("GMAIL_APP_PW")
