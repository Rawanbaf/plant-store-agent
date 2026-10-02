from flask import Flask, render_template, request, redirect, url_for
import json
import os
from datetime import datetime
from langgraph.types import Command

from main import app as graph_app
from rag_chat import ask_question

from functools import wraps
from flask import Response
from dotenv import load_dotenv

load_dotenv()

DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD")


def check_auth(username, password):
    return username == DASHBOARD_USERNAME and password == DASHBOARD_PASSWORD


def authenticate():
    return Response(
        "Login required", 401,
        {"WWW-Authenticate": "Basic realm='Login Required'"}
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

app = Flask(__name__)

COMPLAINTS_FILE = "complaints.json"
RESOLVED_STATUSES = ["Restocked", "Replacement Sent", "Refund Issued", "Refund Denied", "Not Available"]

TEAM_STATUS_OPTIONS = {
    "stock_team": ["In Progress", "Restocked", "Not Available"],
    "quality_team": ["In Progress", "Replacement Sent", "Refund Issued"],
    "finance_team": ["In Progress", "Refund Issued", "Refund Denied"],
}

TEAM_LABELS = {
    "stock_team": "Stock",
    "quality_team": "Quality",
    "finance_team": "Finance",
}


def load_complaints():
    if not os.path.exists(COMPLAINTS_FILE):
        return []
    with open(COMPLAINTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_complaints(complaints):
    with open(COMPLAINTS_FILE, "w", encoding="utf-8") as f:
        json.dump(complaints, f, indent=2, ensure_ascii=False)


def time_ago(created_at_str):
    created_at = datetime.fromisoformat(created_at_str)
    delta = datetime.now() - created_at
    if delta.days > 0:
        return f"{delta.days}d"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours}h"
    minutes = (delta.seconds // 60) % 60
    return f"{minutes}m"


@app.route("/")
@requires_auth
def dashboard():
    complaints = load_complaints()

    team_filter = request.args.get("team", "all")
    status_filter = request.args.get("status", "pending")
    search = request.args.get("search", "").strip().lower()

    filtered = []
    for c in complaints:
        is_resolved = c["status"] in RESOLVED_STATUSES

        if team_filter != "all" and c["team"] != team_filter:
            continue
        if search:
            if search not in c["order_number"].lower() and search not in c["sender_email"].lower():
                continue
        else:
            if status_filter == "pending" and is_resolved:
                continue
            if status_filter == "resolved" and not is_resolved:
                continue
        c["time_ago"] = time_ago(c["created_at"])
        filtered.append(c)

    filtered.sort(key=lambda c: c["created_at"], reverse=True)

    pending_count = sum(1 for c in complaints if c["status"] not in RESOLVED_STATUSES)
    resolved_count = sum(1 for c in complaints if c["status"] in RESOLVED_STATUSES)
    low_confidence_count = sum(1 for c in complaints if c["status"] not in RESOLVED_STATUSES and c["confidence"] < 0.7)

    return render_template(
        "dashboard.html",
        complaints=filtered,
        team_status_options=TEAM_STATUS_OPTIONS,
        team_labels=TEAM_LABELS,
        pending_count=pending_count,
        resolved_count=resolved_count,
        low_confidence_count=low_confidence_count,
        team_filter=team_filter,
        status_filter=status_filter,
        search=search,
    )


@app.route("/update_status/<complaint_id>", methods=["POST"])
@requires_auth
def update_status(complaint_id):
    print("update_status called for:", complaint_id)
    new_status = request.form.get("status")
    employee_note = request.form.get("employee_note", "").strip()
    print("new_status:", new_status, "| note:", employee_note)

    complaints = load_complaints()
    target_complaint = None
    for c in complaints:
        if c["id"] == complaint_id:
            c["status"] = new_status
            c["employee_note"] = employee_note
            c["resolved_at"] = datetime.now().isoformat()
            target_complaint = c
            break

    save_complaints(complaints)

    if target_complaint:
        thread_id = target_complaint["email_thread_id"]
        print("Resuming graph for thread:", thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = graph_app.invoke(
                Command(resume={"status": new_status, "note": employee_note}),
                config=config,
            )
            print("Resume result:", result)
        except Exception as e:
            print(f"Failed to resume graph for thread {thread_id}: {e}")
    else:
        print("No matching complaint found for id:", complaint_id)

    return redirect(url_for("dashboard", team=request.args.get("team", "all"), status=request.args.get("status", "pending")))


@app.route("/api/ask", methods=["POST"])
@requires_auth
def api_ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return {"answer": "Please enter a question."}

    answer = ask_question(question)
    return {"answer": answer}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)