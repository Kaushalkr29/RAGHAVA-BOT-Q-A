import os
from dotenv import load_dotenv
from google import genai
import json

class Generation:
    def __init__(self,question,context):
        self.question=question
        self.context=context
    def ask_question(self):
        load_dotenv()
        api_key=os.getenv("GEMINI_API_KEY")
        client=genai.Client(api_key=api_key)

        sysprompt = sysprompt = f"""
You are an AI assistant.

Context:
{self.context}

User Question:
{self.question}

Your task:

1. Answer the user's question using ONLY the provided context.

2. If the user requests one or more actions, return ALL required tools.

Available tools:

- create_pdf
- create_text_file
- send_email

Examples

User:
Create a PDF.

Output:

{{
  "response":"...",
  "tools":["create_pdf"],
  "receiver_email":null
}}

User:
Create a PDF and send it to abc@gmail.com

Output:

{{
  "response":"...",
  "tools":["create_pdf","send_email"],
  "receiver_email":"abc@gmail.com"
}}

User:
Create a text file and email it.

Output:

{{
  "response":"...",
  "tools":["create_text_file","send_email"],
  "receiver_email":null
}}

Return ONLY valid JSON.

3. If an email address is present, extract it into receiver_email.
   Otherwise ask for the receiver_email.

Return ONLY valid JSON.

Example:

{{
  "response": "Answer here",
  "tool": "create_pdf",
  "receiver_email": null
}}
"""
        response = client.models.generate_content(
            model= "gemini-3.1-flash-lite",
            contents=sysprompt
        )
        return json.loads(response.text)