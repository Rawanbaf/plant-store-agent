# Plant Store Complaint Routing Agent

An agentic system that reads customer complaint emails from a real Gmail inbox, understands them, routes them to the right team, and keeps a human in the loop until the customer gets a reply.

Built with LangGraph, OpenAI structured outputs, ChromaDB, Flask, and the Gmail and Slack APIs.

## How it works

```
Customer email
   ↓
Extract category, item, order number, summary, confidence (LLM + Pydantic schema)
   ↓
Not a complaint or low confidence?  →  logged for review, never auto processed
Order number missing?               →  ask the customer by email, pause, resume on reply
   ↓
Route to Stock, Quality, or Finance team (rule based)
   ↓
Save ticket to dashboard and notify the team on Slack
   ↓
Pause until an employee picks an outcome
   ↓
Reply sent to the customer in the same email thread
```

## Key design decisions

* **Humans decide, AI assists.** The employee always chooses the outcome. The AI only writes a personalized reply when the employee adds a note. Otherwise a standard template is sent.
* **LLMs only where language understanding is needed.** Routing is a lookup table, Slack alerts are plain HTTP, and standard replies are templates.
* **Uncertainty is handled explicitly.** Emails that are not complaints, or where the model is unsure, are logged for review instead of being routed wrongly.
* **Durable pause and resume.** The graph is checkpointed in SQLite with two interrupt points: waiting for a customer reply and waiting for an employee decision.
* **Replies stay in the same Gmail thread** using proper reply headers.

## Features

* Real Gmail integration with OAuth, polling every 30 seconds
* Structured extraction with a validated Pydantic schema
* Automatic follow up when an order number is missing
* Per team Slack notifications with a link to the ticket
* Dashboard with filters, search, and per team status options
* Ticket page with similar past tickets (semantic search), customer history, and a collapsible agent timeline
* Analytics page with resolution time, confidence rate, and volume charts
* Assistant chat on every page, answering from past complaints and store policy (RAG)
* Password protected dashboard

## Tech stack

Python, LangGraph, OpenAI API, Pydantic, ChromaDB, sentence-transformers, Flask, Chart.js, Gmail API, Slack webhooks

## Project structure

```
main.py             LangGraph pipeline and the email checking loop
flask_app.py        Dashboard backend and routes
extractor.py        LLM extraction with a Pydantic schema
router.py           Category to team routing
notifier.py         Slack notifications
ask_customer.py     Order number request email
status_update.py    Status reply templates and AI personalized replies
email_reader.py     Gmail reading, sending, and threading
rag_chat.py         Retrieval and similar ticket search
rag_indexer.py      Indexes the knowledge base
knowledge_base.txt  Store policy and plant care content
templates/          HTML pages
static/             Styles
```

## Setup

1. Install dependencies

```
python -m pip install -r requirements.txt
```

2. Gmail: create a Google Cloud project, enable the Gmail API, configure the OAuth consent screen, add your test account as a test user, create a Desktop OAuth client, and save the file as `credentials.json` in the project folder.

3. Slack: create three channels (stock, quality, finance) and an incoming webhook for each.

4. Create a `.env` file:

```
OPENAI_API_KEY=your_key
SLACK_STOCK_WEBHOOK=your_webhook
SLACK_QUALITY_WEBHOOK=your_webhook
SLACK_FINANCE_WEBHOOK=your_webhook
DASHBOARD_USERNAME=your_username
DASHBOARD_PASSWORD=your_password
```

5. Index the knowledge base once

```
python rag_indexer.py
```

## Running

Open two terminals and leave both running.

```
python main.py
```

```
python flask_app.py
```

The first run of `main.py` opens a browser to authorize Gmail. Then open http://localhost:5000 and sign in.

## Notes and limitations

* Runs locally. The embedding model is larger than the memory of free hosting tiers.
* Data is stored in a JSON file and SQLite. A hosted database would be the next step for production.
* Email is checked by polling. Gmail push notifications would reduce the delay.
* The Google OAuth app is in testing mode, so only listed test accounts can sign in.

