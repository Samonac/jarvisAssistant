"""Email Client for Jarvis Assistant.

Sends emails via SMTP with TLS support. Never sends without explicit
confirmation from the Conversation Manager.
"""

import logging
import smtplib
import socket
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from app.config import Config

logger = logging.getLogger(__name__)

# SMTP timeout in seconds
SMTP_TIMEOUT = 30


@dataclass
class EmailDraft:
    """Represents a composed email ready for sending."""

    to: str
    subject: str
    body: str
    from_address: str


class EmailClient:
    """Sends emails via SMTP with TLS support.

    Attributes:
        host: SMTP server hostname.
        port: SMTP server port.
        username: SMTP authentication username.
        password: SMTP authentication password.
        from_address: Sender email address.
    """

    def __init__(self, config: Config):
        self.host: Optional[str] = config.smtp_host
        self.port: int = config.smtp_port
        self.username: Optional[str] = config.smtp_username
        self.password: Optional[str] = config.smtp_password
        self.from_address: Optional[str] = config.smtp_from_address

    def is_configured(self) -> bool:
        """Check if SMTP settings are present and sufficient for sending."""
        return bool(
            self.host
            and self.username
            and self.password
            and self.from_address
        )

    def compose_draft(self, to: str, subject: str, body: str) -> EmailDraft:
        """Create an email draft for user confirmation.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body: Email body text.

        Returns:
            An EmailDraft instance ready for sending.
        """
        return EmailDraft(
            to=to,
            subject=subject,
            body=body,
            from_address=self.from_address or "",
        )

    def send(self, draft: EmailDraft) -> dict:
        """Send an email.

        Args:
            draft: The EmailDraft to send.

        Returns:
            Dict with 'success' (bool) and 'message' (str) keys.
        """
        if not self.is_configured():
            return {
                "success": False,
                "message": "Email sending is not configured. Please set the SMTP environment variables.",
            }

        try:
            # Build the email message
            msg = MIMEMultipart()
            msg["From"] = draft.from_address
            msg["To"] = draft.to
            msg["Subject"] = draft.subject
            msg.attach(MIMEText(draft.body, "plain"))

            # Connect and send
            if self.port == 465:
                # SSL connection
                server = smtplib.SMTP_SSL(
                    self.host, self.port, timeout=SMTP_TIMEOUT
                )
            else:
                # STARTTLS connection
                server = smtplib.SMTP(self.host, self.port, timeout=SMTP_TIMEOUT)
                server.starttls()

            server.login(self.username, self.password)
            server.sendmail(draft.from_address, [draft.to], msg.as_string())
            server.quit()

            logger.info("Email sent successfully to %s", draft.to)
            return {"success": True, "message": "Email sent successfully."}

        except smtplib.SMTPAuthenticationError as e:
            logger.error("SMTP authentication failure: %s", e)
            return {
                "success": False,
                "message": "SMTP authentication failed. Please verify your email credentials.",
            }
        except smtplib.SMTPRecipientsRefused as e:
            logger.error("SMTP recipient refused: %s", e)
            return {
                "success": False,
                "message": f"The recipient address was rejected: {draft.to}",
            }
        except smtplib.SMTPConnectError as e:
            logger.error("SMTP connection error: %s", e)
            return {
                "success": False,
                "message": "Could not connect to the mail server. Please verify SMTP settings.",
            }
        except (socket.timeout, TimeoutError) as e:
            logger.error("SMTP timeout: %s", e)
            return {
                "success": False,
                "message": "The email send operation timed out.",
            }
        except (socket.gaierror, OSError) as e:
            logger.error("SMTP network error: %s", e)
            return {
                "success": False,
                "message": "Network error while connecting to the mail server.",
            }
        except smtplib.SMTPException as e:
            logger.error("SMTP error: %s", e)
            return {
                "success": False,
                "message": f"Email sending failed: {e}",
            }
        except Exception as e:
            logger.error("Unexpected email error: %s", e)
            return {
                "success": False,
                "message": f"An unexpected error occurred while sending the email: {e}",
            }
