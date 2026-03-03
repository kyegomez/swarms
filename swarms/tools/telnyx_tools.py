"""Telnyx communication tools for Swarms agents.

This module provides ready-to-use tool functions for SMS messaging and
voice calls via the Telnyx API. These tools can be passed directly to
Swarms Agent instances for telephony capabilities.

Requirements:
    pip install telnyx

Setup:
    Set the following environment variables:
        TELNYX_API_KEY: Your Telnyx API key
        TELNYX_FROM_NUMBER: Your Telnyx phone number in E.164 format

Usage:
    from swarms import Agent
    from swarms.tools.telnyx_tools import telnyx_send_sms, telnyx_make_call

    agent = Agent(
        agent_name="Communication-Agent",
        model_name="gpt-4o",
        tools=[telnyx_send_sms, telnyx_make_call],
    )

    agent.run("Send an SMS to +15551234567 saying 'Hello from Swarms!'")
"""

import json
import os
from typing import Optional

from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="telnyx_tools")


def _get_telnyx_client():
    """Initialize and return the Telnyx client.

    Returns:
        The telnyx module configured with API key.

    Raises:
        ImportError: If the telnyx package is not installed.
        ValueError: If the TELNYX_API_KEY environment variable is not set.
    """
    try:
        import telnyx
    except ImportError:
        raise ImportError(
            "The 'telnyx' package is required for Telnyx tools. "
            "Install it with: pip install telnyx"
        )

    api_key = os.environ.get("TELNYX_API_KEY")
    if not api_key:
        raise ValueError(
            "TELNYX_API_KEY environment variable is not set. "
            "Get your API key from https://portal.telnyx.com"
        )

    telnyx.api_key = api_key
    return telnyx


def telnyx_send_sms(
    to: str,
    body: str,
    from_number: Optional[str] = None,
    messaging_profile_id: Optional[str] = None,
) -> str:
    """Send an SMS text message using the Telnyx API.

    This tool sends a text message to a specified phone number via
    the Telnyx messaging platform. All phone numbers must be in
    E.164 format (e.g., '+15551234567').

    Args:
        to (str): The destination phone number in E.164 format
            (e.g., '+15551234567').
        body (str): The text content of the message to send.
        from_number (Optional[str]): The sender phone number in E.164
            format. If not provided, uses the TELNYX_FROM_NUMBER
            environment variable.
        messaging_profile_id (Optional[str]): The Telnyx messaging
            profile ID. If not provided, uses the
            TELNYX_MESSAGING_PROFILE_ID environment variable.

    Returns:
        str: A JSON string containing the result with message ID
            and status, or an error message.

    Examples:
        >>> telnyx_send_sms("+15551234567", "Hello from Swarms!")
        '{"success": true, "message_id": "msg_xxx", "to": "+15551234567", "status": "queued"}'

        >>> telnyx_send_sms("+15551234567", "Hello!", from_number="+15559876543")
        '{"success": true, "message_id": "msg_xxx", "to": "+15551234567", "status": "queued"}'
    """
    try:
        telnyx = _get_telnyx_client()

        sender = from_number or os.environ.get("TELNYX_FROM_NUMBER")
        if not sender:
            return json.dumps({
                "success": False,
                "error": (
                    "No sender phone number provided. "
                    "Either pass 'from_number' parameter or set "
                    "TELNYX_FROM_NUMBER environment variable."
                ),
            })

        profile_id = messaging_profile_id or os.environ.get(
            "TELNYX_MESSAGING_PROFILE_ID"
        )

        message_params = {
            "from_": sender,
            "to": to,
            "text": body,
        }

        if profile_id:
            message_params["messaging_profile_id"] = profile_id

        logger.info(f"Sending SMS to {to}")
        message = telnyx.Message.create(**message_params)

        result = {
            "success": True,
            "message_id": str(message.id),
            "to": to,
            "status": str(message.status),
        }

        logger.info(
            f"SMS sent successfully to {to}. "
            f"Message ID: {message.id}"
        )
        return json.dumps(result)

    except ImportError as e:
        logger.error(f"Import error: {e}")
        return json.dumps({"success": False, "error": str(e)})
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return json.dumps({"success": False, "error": str(e)})
    except Exception as e:
        logger.error(f"Error sending SMS: {e}")
        return json.dumps({
            "success": False,
            "error": f"Failed to send SMS: {str(e)}",
        })


def telnyx_make_call(
    to: str,
    from_number: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> str:
    """Make a voice phone call using the Telnyx Call Control API.

    This tool initiates a phone call to a specified number using
    Telnyx Call Control. The call will be connected and can be
    managed using the returned call_control_id.

    Args:
        to (str): The destination phone number in E.164 format
            (e.g., '+15551234567').
        from_number (Optional[str]): The caller phone number in E.164
            format. If not provided, uses the TELNYX_FROM_NUMBER
            environment variable.
        webhook_url (Optional[str]): A URL to receive call status
            webhook events. Useful for monitoring call progress.

    Returns:
        str: A JSON string containing the result with call_control_id,
            or an error message.

    Examples:
        >>> telnyx_make_call("+15551234567")
        '{"success": true, "call_control_id": "cc_xxx", "to": "+15551234567"}'

        >>> telnyx_make_call("+15551234567", webhook_url="https://example.com/webhook")
        '{"success": true, "call_control_id": "cc_xxx", "to": "+15551234567"}'
    """
    try:
        telnyx = _get_telnyx_client()

        caller = from_number or os.environ.get("TELNYX_FROM_NUMBER")
        if not caller:
            return json.dumps({
                "success": False,
                "error": (
                    "No caller phone number provided. "
                    "Either pass 'from_number' parameter or set "
                    "TELNYX_FROM_NUMBER environment variable."
                ),
            })

        call_params = {
            "from_": caller,
            "to": to,
        }

        if webhook_url:
            call_params["webhook_url"] = webhook_url

        logger.info(f"Initiating call to {to}")
        call = telnyx.Call.create(**call_params)

        result = {
            "success": True,
            "call_control_id": str(call.call_control_id),
            "to": to,
        }

        logger.info(
            f"Call initiated to {to}. "
            f"Call Control ID: {call.call_control_id}"
        )
        return json.dumps(result)

    except ImportError as e:
        logger.error(f"Import error: {e}")
        return json.dumps({"success": False, "error": str(e)})
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return json.dumps({"success": False, "error": str(e)})
    except Exception as e:
        logger.error(f"Error making call: {e}")
        return json.dumps({
            "success": False,
            "error": f"Failed to make call: {str(e)}",
        })


def telnyx_hangup_call(call_control_id: str) -> str:
    """Hang up an active phone call using the Telnyx Call Control API.

    This tool ends an active call that was previously initiated
    with telnyx_make_call. You need the call_control_id that was
    returned when the call was created.

    Args:
        call_control_id (str): The call control ID of the active
            call to hang up. This is returned by telnyx_make_call.

    Returns:
        str: A JSON string confirming the hangup or an error message.

    Examples:
        >>> telnyx_hangup_call("cc_abc123def456")
        '{"success": true, "call_control_id": "cc_abc123def456", "action": "hangup"}'
    """
    try:
        telnyx = _get_telnyx_client()

        logger.info(
            f"Hanging up call with ID: {call_control_id}"
        )
        call = telnyx.Call.retrieve(call_control_id)
        call.hangup()

        result = {
            "success": True,
            "call_control_id": call_control_id,
            "action": "hangup",
        }

        logger.info(f"Call {call_control_id} hung up successfully")
        return json.dumps(result)

    except ImportError as e:
        logger.error(f"Import error: {e}")
        return json.dumps({"success": False, "error": str(e)})
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return json.dumps({"success": False, "error": str(e)})
    except Exception as e:
        logger.error(f"Error hanging up call: {e}")
        return json.dumps({
            "success": False,
            "error": f"Failed to hang up call: {str(e)}",
        })


def telnyx_lookup_number(phone_number: str) -> str:
    """Look up information about a phone number using Telnyx.

    This tool queries the Telnyx Number Lookup API to get carrier,
    caller name, and other information about a phone number.

    Args:
        phone_number (str): The phone number to look up in E.164
            format (e.g., '+15551234567').

    Returns:
        str: A JSON string containing the lookup results including
            carrier info, caller name, and other available data.

    Examples:
        >>> telnyx_lookup_number("+15551234567")
        '{"success": true, "phone_number": "+15551234567", "carrier": {...}}'
    """
    try:
        telnyx = _get_telnyx_client()

        logger.info(f"Looking up number: {phone_number}")
        lookup = telnyx.NumberLookup.retrieve(phone_number)

        result = {
            "success": True,
            "phone_number": phone_number,
            "carrier": (
                lookup.carrier if hasattr(lookup, "carrier") else None
            ),
            "caller_name": (
                lookup.caller_name
                if hasattr(lookup, "caller_name")
                else None
            ),
            "portability": (
                lookup.portability
                if hasattr(lookup, "portability")
                else None
            ),
        }

        logger.info(f"Number lookup completed for {phone_number}")
        return json.dumps(result, default=str)

    except ImportError as e:
        logger.error(f"Import error: {e}")
        return json.dumps({"success": False, "error": str(e)})
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return json.dumps({"success": False, "error": str(e)})
    except Exception as e:
        logger.error(f"Error looking up number: {e}")
        return json.dumps({
            "success": False,
            "error": f"Failed to look up number: {str(e)}",
        })
