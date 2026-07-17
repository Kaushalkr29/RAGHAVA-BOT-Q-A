import json
from agent.tools import Agent
from generation.gemini_generation import Generation
from src.retrieval import retrieve
from src.vectorizer import Vectorizer
import os

LOG_FILE = "agent/logs.json"

TOOL_METADATA = {
    "create_pdf": {
        "attributes": ["content"],
        "description": "Creates a PDF file using the provided content/context data. The content is written into a PDF document and the generated PDF file path is returned."
    },
    "create_text_file": {
        "attributes": ["content"],
        "description": "Creates a text file using the provided content/context data. The content is saved into a .txt file and the generated text file path is returned."
    },
    "send_email": {
        "attributes": ["receiver_email", "content"],
        "description": "Sends an email to the provided receiver email address using the content/context data as the email body."
    }
}


def save_tool_log(tool_name):
    """
    Save only the tool that was actually used.
    """

    if tool_name not in TOOL_METADATA:
        return

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = {}
    else:
        logs = {}

    logs[tool_name] = TOOL_METADATA[tool_name]

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4)

    with open("agent/logs.json", "r", encoding="utf-8") as f:
        logs = json.load(f)

agent = Agent(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
)

def execute_tool(tool_name, content, receiver_email=None):

    if tool_name is None:
        return None

    save_tool_log(tool_name)

    if tool_name == "create_pdf":
        return {
    "status": "success",
    "message": agent.create_pdf(content)
}

    elif tool_name == "create_text_file":
        return {
    "status": "success",
    "message": agent.create_text_file(content)
}

    elif tool_name == "send_email":

        if not receiver_email:
            return {
                "status": "need_email",
                "message": "Please provide the recipient email."
            }

        return {
    "status": "success",
    "message": agent.send_email(receiver_email,content)
}


def process_user_prompt(user_prompt,context):

    generation = Generation(user_prompt,context)

    result = generation.ask_question()

    """
    Expected format returned by Generation():

    {
        "response": "...",
        "tool": "create_pdf",
        "receiver_email": "abc@gmail.com"
    }
    """

    response = result["response"]
    tools = result.get("tools", [])

    for tool in tools:

        tool_result = execute_tool(
            tool,
            response,
            receiver_email=None
        )

        print(tool_result)

    return {
        "response": response,
        "tool_result": tool_result
    }


if __name__ == "__main__":

    while True:

        prompt = input("User : ")
        r=retrieve(prompt,Vectorizer().dense_vector())
        context,q=r.top_k()

        if prompt.lower() == "exit":
            break

        result = process_user_prompt(prompt, context)

        print("\nAssistant:\n")
        print(result["response"])

        if result["tool_result"]:

            if result["tool_result"].get("status") == "need_email":

                email = input("\nEnter recipient email: ")

                execute_tool(
                    "send_email",
                    result["response"],
                    email
                )

                print("Email sent successfully.")

            else:

                print(result["tool_result"])

