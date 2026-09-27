import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ComplaintInfo(BaseModel):
    category: str = Field(description="One of: stock, bad_quality, finance, not_a_complaint")
    summary: str = Field(description="A short summary of the customer's issue")
    item: str = Field(description="The product mentioned, if any")
    confidence: float = Field(description="Confidence score between 0 and 1")
    order_number: str = Field(description="The order number if mentioned, otherwise 'not provided'")

SYSTEM_PROMPT = """You are an assistant for a plant store's customer service team.
You will receive a customer complaint email. Extract the following:

- category: must be exactly one of "stock", "bad_quality", or "finance"
  - stock: item unavailable, out of stock, availability questions
  - bad_quality: item arrived damaged, wrong, or defective
  - finance: refunds, billing, payment issues

- summary: one short sentence capturing the core issue, in your own words

- item: the specific product mentioned (e.g. "monstera plant"), or "unspecified" if none is mentioned

- order_number: the order number if the customer mentions one (e.g. "order #1234"), otherwise exactly "not provided"

- confidence: your own estimate (0 to 1) of how clear-cut this classification is.
  Use lower values if the email is ambiguous or mentions multiple issues.

  - not_a_complaint: use this when the email is NOT a genuine customer complaint about a plant store order or product. Examples include:
  - Automated system notifications (account security alerts, password changes, login notifications)
  - Marketing emails, newsletters, or promotional content
  - Spam or unrelated advertisements
  - General inquiries with no actual issue (e.g. "just saying hi")
  - Emails from services unrelated to the plant store (e.g. software subscriptions, other companies)
  
  If you classify something as not_a_complaint, set item to "unspecified" and order_number to "not provided", and briefly describe what the email actually is in the summary.
"""


def extract_complaint_info(email_text: str) -> ComplaintInfo:
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email_text},
        ],
        text_format=ComplaintInfo,
    )
    return response.output_parsed


