from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os
import re
import json
import uuid
import time
from datetime import datetime

from extractor import extract_complaint_info
from router import route_to_team
from notifier import notify_team
from ask_customer import generate_order_number_request
from status_update import get_status_message
from email_reader import get_gmail_service, get_new_emails, mark_as_processed, send_email


PAUSED_THREADS_FILE = "paused_threads.txt"
COMPLAINTS_FILE = "complaints.json"


class State(TypedDict):
    email_id: str
    email_text: str
    sender_email: str
    email_thread_id: str
    message_id_header: str
    original_subject: str
    category: str
    summary: str
    item: str
    confidence: float
    order_number: str
    team: str
    employee_status: str
    employee_note: str


def extract_email_address(sender: str) -> str:
    match = re.search(r'<(.+?)>', sender)
    return match.group(1) if match else sender


def extractor_node(state: State):
    result = extract_complaint_info(state["email_text"])
    return {
        "category": result.category,
        "summary": result.summary,
        "item": result.item,
        "confidence": result.confidence,
        "order_number": result.order_number,
    }


def send_request_node(state: State):
    message = generate_order_number_request(state["email_text"])

    service = get_gmail_service()
    send_email(
        service,
        to=state["sender_email"],
        subject="Re: " + state["original_subject"],
        body=message,
        thread_id=state["email_thread_id"],
        in_reply_to=state["message_id_header"],
    )
    print(f"Sent order-number request to {state['sender_email']}")
    return {}


def wait_for_reply_node(state: State):
    customer_reply = interrupt({"type": "await_customer_reply", "message_sent": "waiting for reply"})
    combined_text = state["email_text"] + "\n\nCustomer's follow-up reply: " + customer_reply
    return {"email_text": combined_text}


def log_ignored_node(state: State):
    with open("ignored_emails.log", "a", encoding="utf-8") as f:
        f.write(f"{state['email_id']} | {state['summary']}\n")
    print("Logged as ignored (not a complaint).")
    return {}


def router_node(state: State):
    team = route_to_team(state["category"])
    return {"team": team}


def notify_node(state: State):
    notify_team(
        team=state["team"],
        category=state["category"],
        summary=state["summary"],
        item=state["item"],
        order_number=state["order_number"],
        full_email=state["email_text"],
        dashboard_link="http://localhost:5000",
    )
    return {}


def load_complaints():
    if not os.path.exists(COMPLAINTS_FILE):
        return []
    with open(COMPLAINTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_complaints(complaints):
    with open(COMPLAINTS_FILE, "w", encoding="utf-8") as f:
        json.dump(complaints, f, indent=2, ensure_ascii=False)


def save_to_dashboard_node(state: State):
    complaints = load_complaints()
    complaints.append({
        "id": str(uuid.uuid4()),
        "email_thread_id": state["email_thread_id"],
        "team": state["team"],
        "category": state["category"],
        "item": state["item"],
        "order_number": state["order_number"],
        "summary": state["summary"],
        "full_email": state["email_text"],
        "confidence": state["confidence"],
        "sender_email": state["sender_email"],
        "message_id_header": state["message_id_header"],
        "original_subject": state["original_subject"],
        "status": "pending",
        "created_at": datetime.now().isoformat(),
    })
    save_complaints(complaints)
    return {}


def wait_for_employee_node(state: State):
    decision = interrupt({"type": "await_employee_status"})
    return {
        "employee_status": decision["status"],
        "employee_note": decision.get("note", ""),
    }


def send_status_email_node(state: State):
    message = get_status_message(
        status=state["employee_status"],
        original_issue=state["email_text"],
        employee_note=state.get("employee_note", ""),
    )

    service = get_gmail_service()
    send_email(
        service,
        to=state["sender_email"],
        subject="Re: " + state["original_subject"],
        body=message,
        thread_id=state["email_thread_id"],
        in_reply_to=state["message_id_header"],
    )
    print(f"Sent status update to {state['sender_email']}")
    return {}


def route_after_extraction(state: State):
    if state["confidence"] < 0.6:
        return "log_ignored"

    if state["category"] == "not_a_complaint":
        return "log_ignored"

    if state["category"] in ("bad_quality", "finance") and state["order_number"] == "not provided":
        return "ask_customer"

    return "router"


graph = StateGraph(State)
graph.add_node("extractor", extractor_node)
graph.add_node("send_request", send_request_node)
graph.add_node("wait_for_reply", wait_for_reply_node)
graph.add_node("log_ignored", log_ignored_node)
graph.add_node("router", router_node)
graph.add_node("save_to_dashboard", save_to_dashboard_node)
graph.add_node("notify", notify_node)
graph.add_node("wait_for_employee", wait_for_employee_node)
graph.add_node("send_status_email", send_status_email_node)

graph.set_entry_point("extractor")
graph.add_conditional_edges("extractor", route_after_extraction, {
    "log_ignored": "log_ignored",
    "ask_customer": "send_request",
    "router": "router",
})
graph.add_edge("send_request", "wait_for_reply")
graph.add_edge("wait_for_reply", "extractor")
graph.add_edge("log_ignored", END)
graph.add_edge("router", "save_to_dashboard")
graph.add_edge("save_to_dashboard", "notify")
graph.add_edge("notify", "wait_for_employee")
graph.add_edge("wait_for_employee", "send_status_email")
graph.add_edge("send_status_email", END)

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

app = graph.compile(checkpointer=checkpointer)


def load_paused_threads():
    if not os.path.exists(PAUSED_THREADS_FILE):
        return set()
    with open(PAUSED_THREADS_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def mark_thread_paused(thread_id):
    with open(PAUSED_THREADS_FILE, "a") as f:
        f.write(thread_id + "\n")


def unmark_thread_paused(thread_id):
    paused = load_paused_threads()
    paused.discard(thread_id)
    with open(PAUSED_THREADS_FILE, "w") as f:
        for t in paused:
            f.write(t + "\n")


if __name__ == "__main__":
    service = get_gmail_service()

    print("Email checker running. Checking every 30 seconds. Press Ctrl+C to stop.")

    while True:
        new_emails = get_new_emails(service)
        paused_threads = load_paused_threads()

        if new_emails:
            print(f"Found {len(new_emails)} new email(s) to process")

        for email in new_emails:
            thread_id = email["thread_id"]
            config = {"configurable": {"thread_id": thread_id}}

            if thread_id in paused_threads:
                print(f"\nResuming paused conversation (thread {thread_id})...")
                result = app.invoke(Command(resume=email["body"]), config=config)
            else:
                print(f"\nProcessing new email from {email['sender']}...")
                result = app.invoke({
                    "email_id": email["id"],
                    "email_text": email["body"],
                    "sender_email": extract_email_address(email["sender"]),
                    "email_thread_id": email["thread_id"],
                    "message_id_header": email["message_id_header"],
                    "original_subject": email["subject"],
                }, config=config)

            print("Result:", result)

            if "__interrupt__" in result:
                interrupt_info = result["__interrupt__"][0].value
                if interrupt_info.get("type") == "await_customer_reply":
                    mark_thread_paused(thread_id)
                    mark_as_processed(email["id"])
                    print("Paused — waiting for customer reply.")
                else:
                    mark_as_processed(email["id"])
                    print("Paused — waiting for employee status update.")
            else:
                if thread_id in paused_threads:
                    unmark_thread_paused(thread_id)
                mark_as_processed(email["id"])
                print(f"Marked {email['id']} as processed")

        time.sleep(30)