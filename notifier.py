import os
import requests
from dotenv import load_dotenv

load_dotenv()

TEAM_WEBHOOKS = {
    "stock_team": os.getenv("SLACK_STOCK_WEBHOOK"),
    "quality_team": os.getenv("SLACK_QUALITY_WEBHOOK"),
    "finance_team": os.getenv("SLACK_FINANCE_WEBHOOK"),
}


def notify_team(team: str, category: str, summary: str, item: str, order_number: str, full_email: str, dashboard_link: str):
    webhook_url = TEAM_WEBHOOKS.get(team)

    if not webhook_url:
        print(f"No webhook found for team: {team}")
        return

    message = {
        "text": f"New ticket assigned — <{dashboard_link}|View details>"
    }

    response = requests.post(webhook_url, json=message)
    return response.status_code