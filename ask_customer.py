import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You write short, polite customer service emails for a plant store.
You will be given a customer's complaint. Write a brief email asking them to reply with their order number so the team can look into their issue.
Reference their specific issue briefly so it doesn't feel like a generic message. Keep it under 4 sentences.
Start with "Dear Customer," and sign off as "Plant Store Support Team" — do not use placeholder names like [Customer's Name] or [Your Name]."""


def generate_order_number_request(email_text: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email_text},
        ],
    )
    return response.output_text


if __name__ == "__main__":
    sample = "My monstera arrived with broken leaves and a cracked pot."
    print(generate_order_number_request(sample))