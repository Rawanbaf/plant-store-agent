from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import base64
import os
import pickle
import re
from email.mime.text import MIMEText

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.send']
PROCESSED_FILE = 'processed_ids.txt'
TOKEN_FILE = 'token.pickle'

IGNORED_SENDERS = [
    'no-reply@accounts.google.com',
    'noreply@google.com',
    '@google.com',
    'plantstoreagent@gmail.com',
]


def get_gmail_service():
    creds = None

    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)


def load_processed_ids():
    if not os.path.exists(PROCESSED_FILE):
        return set()
    with open(PROCESSED_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def mark_as_processed(email_id):
    with open(PROCESSED_FILE, 'a') as f:
        f.write(email_id + '\n')


def is_ignored_sender(sender: str) -> bool:
    sender_lower = sender.lower()
    return any(ignored in sender_lower for ignored in IGNORED_SENDERS)


def clean_reply_text(text):
    match = re.search(r'\nOn .+ wrote:', text)
    if match:
        return text[:match.start()].strip()
    
    match = re.search(r'\n.*تمت كتابة ما يلي بواسطة.*', text)
    if match:
        return text[:match.start()].strip()
    
    return text.strip()


def get_new_emails(service, max_results=10):
    processed_ids = load_processed_ids()

    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])

    new_emails = []

    for msg in messages:
        if msg['id'] in processed_ids:
            continue

        full_msg = service.users().messages().get(userId='me', id=msg['id']).execute()

        headers = full_msg['payload']['headers']
        sender = next(h['value'] for h in headers if h['name'] == 'From')
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '(no subject)')
        message_id_header = next((h['value'] for h in headers if h['name'] == 'Message-ID'), None)

        if is_ignored_sender(sender):
            mark_as_processed(msg['id'])
            continue

        body_text = extract_body(full_msg['payload'])

        new_emails.append({
            "id": msg['id'],
            "thread_id": full_msg['threadId'],
            "message_id_header": message_id_header,
            "sender": sender,
            "subject": subject,
            "body": clean_reply_text(body_text) if body_text else full_msg.get('snippet', '')
        })

    return new_emails


def extract_body(payload):
    if 'parts' in payload:
        for part in payload['parts']:
            if part.get('mimeType') == 'text/plain' and 'data' in part.get('body', {}):
                data = part['body']['data']
                return base64.urlsafe_b64decode(data).decode('utf-8')
        for part in payload['parts']:
            result = extract_body(part)
            if result:
                return result
    else:
        if payload.get('mimeType') == 'text/plain' and 'data' in payload.get('body', {}):
            data = payload['body']['data']
            return base64.urlsafe_b64decode(data).decode('utf-8')
    return None


def send_email(service, to, subject, body, thread_id=None, in_reply_to=None):
    message = MIMEText(body)
    message['to'] = to
    message['subject'] = subject

    if in_reply_to:
        message['In-Reply-To'] = in_reply_to
        message['References'] = in_reply_to

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    email_body = {'raw': raw}
    if thread_id:
        email_body['threadId'] = thread_id

    sent = service.users().messages().send(
        userId='me',
        body=email_body
    ).execute()

    return sent


if __name__ == "__main__":
    service = get_gmail_service()
    emails = get_new_emails(service)
    print(f"Found {len(emails)} new email(s)")
    for email in emails:
        print(email)