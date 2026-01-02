"""
Google Calendar Agent

An agent that can create and read events in Google Calendar using the Google Calendar API.
Requires OAuth2 authentication with Google Calendar API access.

HOW TO GET GOOGLE CALENDAR API ACCESS TOKEN:
===========================================

Method 1: OAuth2 Flow (Recommended for user calendars)
-------------------------------------------------------
1. Go to Google Cloud Console: https://console.cloud.google.com/
2. Create a new project or select an existing one
3. Enable Google Calendar API:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Calendar API"
   - Click "Enable"
4. Create OAuth2 credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop app" or "Web application"
   - Download the credentials JSON file
5. Install required packages:
   pip install google-auth google-auth-oauthlib google-auth-httplib2
6. Use the OAuth2 flow to get an access token:
   - Run the OAuth2 flow to get user consent
   - Save the access token and refresh token
   - Access tokens expire (usually 1 hour), use refresh token to get new ones

Quick OAuth2 example:
```python
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os

SCOPES = ['https://www.googleapis.com/auth/calendar']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'

def get_access_token():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    return creds.token

# Set the access token
os.environ['GOOGLE_CALENDAR_ACCESS_TOKEN'] = get_access_token()
```

Method 2: Service Account (For server-to-server, own calendar only)
-------------------------------------------------------------------
1. Go to Google Cloud Console
2. Enable Google Calendar API (same as Method 1, step 3)
3. Create Service Account:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service account"
   - Create and download the JSON key file
4. Share your calendar with the service account email
5. Use service account credentials to get access token:
```python
from google.oauth2 import service_account
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/calendar']
SERVICE_ACCOUNT_FILE = 'service-account-key.json'

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
credentials.refresh(Request())
access_token = credentials.token
os.environ['GOOGLE_CALENDAR_ACCESS_TOKEN'] = access_token
```

Method 3: Using gcloud CLI (Quick testing)
------------------------------------------
1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
2. Authenticate: gcloud auth application-default login
3. Get token: gcloud auth print-access-token
4. Set environment variable:
   export GOOGLE_CALENDAR_ACCESS_TOKEN=$(gcloud auth print-access-token)

Note: Access tokens expire. For production, implement token refresh logic.
For more details: https://developers.google.com/calendar/api/guides/auth
"""

from datetime import datetime
from typing import Dict, Optional, Any

import httpx
from swarms import Agent
from voice_agents import StreamingTTSCallback


def create_calendar_event(
    summary: str,
    description: Optional[str] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    location: Optional[str] = None,
    timezone: str = "UTC",
    calendar_id: str = "primary",
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new event in Google Calendar.

    Args:
        summary (str): The title/summary of the event (required).
        description (Optional[str]): A detailed description of the event. Defaults to None.
        start_datetime (Optional[str]): Start date and time in RFC3339 format (e.g., "2024-01-15T10:00:00Z").
            If not provided, defaults to current time. Defaults to None.
        end_datetime (Optional[str]): End date and time in RFC3339 format (e.g., "2024-01-15T11:00:00Z").
            If not provided, defaults to 1 hour after start time. Defaults to None.
        location (Optional[str]): The location of the event. Defaults to None.
        timezone (str): The timezone for the event (e.g., "America/New_York", "UTC"). Defaults to "UTC".
        calendar_id (str): The calendar ID where the event should be created. Defaults to "primary".
        access_token (Optional[str]): OAuth2 access token for Google Calendar API.
            If not provided, will try to get from GOOGLE_CALENDAR_ACCESS_TOKEN environment variable. Defaults to None.

    Returns:
        Dict[str, Any]: The created event data from Google Calendar API, or error information.

    Raises:
        ValueError: If required parameters are missing or invalid.
        httpx.HTTPError: If the API request fails.
    """
    import os

    if not summary:
        raise ValueError(
            "summary is required to create a calendar event"
        )

    if access_token is None:
        access_token = os.getenv("GOOGLE_CALENDAR_ACCESS_TOKEN")

    if not access_token:
        return {
            "error": "Access token is required. Set GOOGLE_CALENDAR_ACCESS_TOKEN environment variable or pass access_token parameter."
        }

    if start_datetime is None:
        start_datetime = datetime.utcnow().isoformat() + "Z"

    if end_datetime is None:
        start_dt = datetime.fromisoformat(
            start_datetime.replace("Z", "+00:00")
        )
        end_dt = start_dt.replace(hour=start_dt.hour + 1)
        end_datetime = end_dt.isoformat().replace("+00:00", "Z")

    event_data = {
        "summary": summary,
        "start": {
            "dateTime": start_datetime,
            "timeZone": timezone,
        },
        "end": {
            "dateTime": end_datetime,
            "timeZone": timezone,
        },
    }

    if description:
        event_data["description"] = description

    if location:
        event_data["location"] = location

    url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                url,
                headers=headers,
                json=event_data,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP error {e.response.status_code}: {e.response.text}",
            "status_code": e.response.status_code,
        }
    except httpx.RequestError as e:
        return {"error": f"Request error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def read_calendar_events(
    calendar_id: str = "primary",
    max_results: int = 10,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    query: Optional[str] = None,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Read events from Google Calendar.

    Args:
        calendar_id (str): The calendar ID to read events from. Defaults to "primary".
        max_results (int): Maximum number of events to return (1-2500). Defaults to 10.
        time_min (Optional[str]): Lower bound (exclusive) for an event's start time in RFC3339 format.
            If not provided, defaults to current time. Defaults to None.
        time_max (Optional[str]): Upper bound (exclusive) for an event's start time in RFC3339 format.
            If not provided, no upper bound is applied. Defaults to None.
        query (Optional[str]): Free text search terms to find events that match these terms.
            Defaults to None.
        access_token (Optional[str]): OAuth2 access token for Google Calendar API.
            If not provided, will try to get from GOOGLE_CALENDAR_ACCESS_TOKEN environment variable. Defaults to None.

    Returns:
        Dict[str, Any]: The events data from Google Calendar API, or error information.

    Raises:
        ValueError: If max_results is out of valid range.
        httpx.HTTPError: If the API request fails.
    """
    import os

    if max_results < 1 or max_results > 2500:
        raise ValueError("max_results must be between 1 and 2500")

    if access_token is None:
        access_token = os.getenv("GOOGLE_CALENDAR_ACCESS_TOKEN")

    if not access_token:
        return {
            "error": "Access token is required. Set GOOGLE_CALENDAR_ACCESS_TOKEN environment variable or pass access_token parameter."
        }

    if time_min is None:
        time_min = datetime.utcnow().isoformat() + "Z"

    url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    params = {
        "maxResults": max_results,
        "timeMin": time_min,
        "singleEvents": "true",
        "orderBy": "startTime",
    }

    if time_max:
        params["timeMax"] = time_max

    if query:
        params["q"] = query

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                url,
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP error {e.response.status_code}: {e.response.text}",
            "status_code": e.response.status_code,
        }
    except httpx.RequestError as e:
        return {"error": f"Request error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def list_calendars(
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List all calendars accessible to the authenticated user.

    Args:
        access_token (Optional[str]): OAuth2 access token for Google Calendar API.
            If not provided, will try to get from GOOGLE_CALENDAR_ACCESS_TOKEN environment variable. Defaults to None.

    Returns:
        Dict[str, Any]: The calendars list from Google Calendar API, or error information.

    Raises:
        httpx.HTTPError: If the API request fails.
    """
    import os

    if access_token is None:
        access_token = os.getenv("GOOGLE_CALENDAR_ACCESS_TOKEN")

    if not access_token:
        return {
            "error": "Access token is required. Set GOOGLE_CALENDAR_ACCESS_TOKEN environment variable or pass access_token parameter."
        }

    url = (
        "https://www.googleapis.com/calendar/v3/users/me/calendarList"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                url,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP error {e.response.status_code}: {e.response.text}",
            "status_code": e.response.status_code,
        }
    except httpx.RequestError as e:
        return {"error": f"Request error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# Create TTS callback for voice output
tts_callback = StreamingTTSCallback(
    voice="alloy",
    model="openai/tts-1",
    stream_mode=True,
)

agent = Agent(
    agent_name="Google-Calendar-Agent",
    agent_description="An agent that can create and read events in Google Calendar using the Google Calendar API",
    model_name="gpt-4o-mini",
    max_loops=3,
    tools=[
        create_calendar_event,
        read_calendar_events,
        list_calendars,
    ],
    system_prompt="""You are a Google Calendar assistant agent. Your primary responsibilities are:

1. Creating calendar events with proper date/time formatting
2. Reading and retrieving calendar events
3. Listing available calendars
4. Providing helpful information about calendar events

When creating events:
- Always use RFC3339 format for dates (e.g., "2024-01-15T10:00:00Z" for UTC or "2024-01-15T10:00:00-05:00" for EST)
- Ensure end time is after start time
- Provide clear summaries and descriptions

When reading events:
- Use appropriate time ranges to filter events
- Format the response in a user-friendly way
- Include all relevant event details

Always handle errors gracefully and provide clear feedback to users about what actions were taken.""",
    dynamic_context_window=True,
    verbose=False,
    print_on=False,
    streaming_on=True,
    streaming_callback=tts_callback,
)


def run_calendar_agent_with_voice(task: str) -> str:
    """
    Run the Google Calendar agent with text-to-speech output.

    Args:
        task (str): The task for the agent to perform (e.g., "Create an event called Team Meeting tomorrow at 2pm").

    Returns:
        str: The agent's response text.

    Example:
        >>> result = run_calendar_agent_with_voice("Show me all events for next week")
        >>> # The agent will speak the response using TTS
    """
    try:
        result = agent.run(task=task, streaming_callback=tts_callback)
        tts_callback.flush()
        return result
    except Exception as e:
        tts_callback.flush()
        raise e
