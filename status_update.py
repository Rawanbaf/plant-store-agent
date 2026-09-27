import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You write short, warm customer service emails for a plant store, informing a customer about an update to their complaint.

You will be given:
- The customer's original issue
- The new status of their complaint
- A note from the employee explaining the decision or any extra context to include

Write a brief email (3-4 sentences) that:
- Clearly communicates the outcome, incorporating the employee's note naturally (do not just repeat it verbatim, write it as a genuine explanation to the customer)
- Matches tone to the status: apologetic and understanding for a denial, warm for a refund or replacement, helpful for stock updates
- References their original issue briefly, so it feels personal, not generic
- Signs off as "Plant Store Support Team"
- Does not use placeholder names like [Customer's Name]
- Starts with "Dear Customer,"
"""


def generate_status_update(original_issue: str, status: str, employee_note: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Original issue: {original_issue}\nNew status: {status}\nEmployee's note: {employee_note}"},
        ],
    )
    return response.output_text


STATIC_STATUS_MESSAGES = {
    # Stock team
    "In Progress": "Dear Customer,\n\nWe wanted to let you know we're actively looking into your request and will follow up with an update soon.\n\nPlant Store Support Team",
    "Restocked": "Dear Customer,\n\nGood news, the item you asked about is back in stock! Feel free to place your order whenever you're ready.\n\nPlant Store Support Team",
    "Not Available": "Dear Customer,\n\nUnfortunately, this item isn't currently available and we don't have a restock date yet. We'll be sure to let you know if that changes.\n\nPlant Store Support Team",

    # Bad Quality team
    "Replacement Sent": "Dear Customer,\n\nYour replacement item is on its way. Thank you for your patience while we resolved this.\n\nPlant Store Support Team",
    "Refund Issued": "Dear Customer,\n\nYour refund has been processed. Thank you for your patience, and we're sorry for the inconvenience.\n\nPlant Store Support Team",

    # Finance team
    "Refund Denied": "Dear Customer,\n\nAfter reviewing your request, we're unable to process a refund at this time. If you have any questions, feel free to reply to this email.\n\nPlant Store Support Team",
}


def get_status_message(status: str, original_issue: str, employee_note: str = "") -> str:
    if employee_note.strip():
        return generate_status_update(original_issue, status, employee_note)
    return STATIC_STATUS_MESSAGES.get(status, f"Dear Customer,\n\nAn update has been made to your request: {status}.\n\nPlant Store Support Team")


