from flask import Flask, render_template, request, redirect, url_for, Response
from functools import wraps
from dotenv import load_dotenv
from datetime import datetime, timedelta
from langgraph.types import Command
import json
import os
import re

load_dotenv()

from main import app as graph_app
from rag_chat import ask_question, find_similar, sync_complaints

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


def format_duration(seconds):
    if seconds is None:
        return "n/a"
    if seconds < 3600:
        return f"{round(seconds / 60)}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


@app.template_filter("clock")
def clock_filter(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%d %b, %I:%M %p")
    except Exception:
        return ""


@app.template_filter("item_name")
def item_name_filter(item):
    if (item or "").strip().lower() in ("", "unspecified", "none", "not provided"):
        return "Item not specified"
    return item


@app.template_filter("order_label")
def order_label_filter(value):
    v = (value or "").strip()
    if v.lower() in ("", "not provided", "none", "unspecified"):
        return "No order number"
    v = re.sub(r"^order\s*", "", v, flags=re.I).lstrip("#").strip()
    return f"Order #{v}"


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
    low_confidence_count = sum(
        1 for c in complaints if c["status"] not in RESOLVED_STATUSES and c["confidence"] < 0.7
    )

    return render_template(
        "dashboard.html",
        complaints=filtered,
        team_labels=TEAM_LABELS,
        pending_count=pending_count,
        resolved_count=resolved_count,
        low_confidence_count=low_confidence_count,
        team_filter=team_filter,
        status_filter=status_filter,
        search=search,
        active_page="dashboard",
    )


@app.route("/ticket/<complaint_id>")
@requires_auth
def ticket_detail(complaint_id):
    complaints = load_complaints()
    complaint = next((c for c in complaints if c["id"] == complaint_id), None)

    if not complaint:
        return redirect(url_for("dashboard"))

    complaint["time_ago"] = time_ago(complaint["created_at"])
    is_resolved = complaint["status"] in RESOLVED_STATUSES

    history = [
        h for h in complaints
        if h["sender_email"] == complaint["sender_email"] and h["id"] != complaint_id
    ]
    history.sort(key=lambda h: h["created_at"], reverse=True)

    candidates = find_similar(complaint, complaints)
    candidates.sort(key=lambda s: 0 if s["status"] in RESOLVED_STATUSES else 1)
    similar_tickets = candidates[:3]

    return render_template(
        "ticket.html",
        c=complaint,
        team_status_options=TEAM_STATUS_OPTIONS,
        team_labels=TEAM_LABELS,
        is_resolved=is_resolved,
        history_recent=history[:3],
        history_total=len(history),
        similar_tickets=similar_tickets,
        events=complaint.get("events", []),
        active_page="ticket",
    )


@app.route("/update_status/<complaint_id>", methods=["POST"])
@requires_auth
def update_status(complaint_id):
    new_status = request.form.get("status")
    employee_note = request.form.get("employee_note", "").strip()

    complaints = load_complaints()
    target = next((c for c in complaints if c["id"] == complaint_id), None)

    if not target or new_status not in TEAM_STATUS_OPTIONS.get(target["team"], []):
        return redirect(url_for("dashboard"))

    target["status"] = new_status
    target["employee_note"] = employee_note
    if new_status in RESOLVED_STATUSES:
        target["resolved_at"] = datetime.now().isoformat()
    save_complaints(complaints)

    config = {"configurable": {"thread_id": target["email_thread_id"]}}
    try:
        graph_app.invoke(
            Command(resume={"status": new_status, "note": employee_note}),
            config=config,
        )
    except Exception as e:
        print(f"Failed to resume graph for thread {target['email_thread_id']}: {e}")

    return redirect(url_for("ticket_detail", complaint_id=complaint_id))


@app.route("/analytics")
@requires_auth
def analytics():
    complaints = load_complaints()
    total = len(complaints)

    durations = []
    for c in complaints:
        if c["status"] in RESOLVED_STATUSES and c.get("resolved_at"):
            delta = datetime.fromisoformat(c["resolved_at"]) - datetime.fromisoformat(c["created_at"])
            durations.append(delta.total_seconds())
    avg_seconds = sum(durations) / len(durations) if durations else None

    high_conf = sum(1 for c in complaints if c["confidence"] >= 0.7)
    high_conf_pct = f"{round(100 * high_conf / total)}%" if total else "n/a"

    needs_review = sum(
        1 for c in complaints if c["status"] not in RESOLVED_STATUSES and c["confidence"] < 0.7
    )

    team_names = ["Stock", "Quality", "Finance"]
    team_keys = ["stock_team", "quality_team", "finance_team"]
    team_values = [sum(1 for c in complaints if c["team"] == k) for k in team_keys]

    volume_labels = []
    volume_values = []
    for i in range(6, -1, -1):
        day = (datetime.now() - timedelta(days=i)).date()
        volume_labels.append(day.strftime("%a"))
        volume_values.append(
            sum(1 for c in complaints if datetime.fromisoformat(c["created_at"]).date() == day)
        )

    return render_template(
        "analytics.html",
        total=total,
        avg_resolution=format_duration(avg_seconds),
        high_conf_pct=high_conf_pct,
        needs_review=needs_review,
        team_names=team_names,
        team_values=team_values,
        volume_labels=volume_labels,
        volume_values=volume_values,
        active_page="analytics",
    )


@app.route("/api/ask", methods=["POST"])
@requires_auth
def api_ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return {"answer": "Please enter a question."}

    try:
        sync_complaints(load_complaints())
    except Exception as e:
        print("Complaint sync failed:", e)

    answer = ask_question(question)
    return {"answer": answer}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)