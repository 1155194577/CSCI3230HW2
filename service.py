import requests
from dotenv import load_dotenv
import os

load_dotenv()

def post(api, data):
    response = requests.post(f"{api}", json=data)
    return response.json()

def get(api):
    response = requests.get(f"{api}")
    return response.json()

def get_summary_from_gpt(text):
    data = { 
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a Summarizer, your job is to summarize text that I give you"},
            {"role": "user", "content": text}
        ]
    }
    
    API = os.getenv("OPENAPI_API")
    KEY = os.getenv("OPENAPI_KEY")
    VALUE = os.getenv("OPENAPI_VALUE")
    headers = {KEY: VALUE}
    
    response = requests.post(f"{API}", json=data, headers=headers)
    response_data = response.json()
    
    if response.status_code == 200:
        return response_data["message"]["content"]
    else:
        print(f"Error: {response_data}")
        return None

def get_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def save_summary_to_md(file_path, summary):
    with open(file_path, 'w') as file:
        file.write(f"# Summary\n\n{summary}")

def main():
    file_path = 'input.txt'
    text = get_text_from_file(file_path)
    summary = get_summary_from_gpt(text)
    
    if summary:
        save_summary_to_md("output.md", summary)

if __name__ == "__main__":
    main()