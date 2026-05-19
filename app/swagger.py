"""Swagger/OpenAPI specification for Jarvis Assistant.

Serves an interactive Swagger UI at /api/docs and the OpenAPI JSON spec at /api/spec.
"""

OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "J.A.R.V.I.S. Assistant API",
        "description": """REST API for the Jarvis personal assistant. Allows interaction from any device on the local network.

## Overview

J.A.R.V.I.S. is a Flask-based AI assistant running on a Raspberry Pi Zero 2W, powered by free-tier LLM APIs (Groq, HuggingFace, Gemini) with automatic failover between providers.

## Authentication

Two authentication methods are supported:

1. **Session cookies** (Web UI) — Log in at `/login` with your IAM credentials. A session cookie is set.
2. **HTTP Basic Auth** (API) — Send `Authorization: Basic <base64(username:password)>` header with every request.

Example with curl:
```bash
curl -u admin:yourpassword http://<pi-ip>:5000/api/sessions
```

Example with Python requests:
```python
import requests
r = requests.get('http://<pi-ip>:5000/api/sessions', auth=('admin', 'yourpassword'))
```

The `/health`, `/api/docs`, `/api/spec`, and `/api/subscribe` endpoints are publicly accessible. All other endpoints require authentication and respect IAM role permissions.

## Real-Time Notifications (SSE)

Subscribe to `GET /api/subscribe` for Server-Sent Events. This is the recommended way for external devices to receive real-time notifications (reminders, scheduled command outputs, proactive messages).

```javascript
const evtSource = new EventSource('http://<pi-ip>:5000/api/subscribe');
evtSource.onmessage = (e) => {
    const data = JSON.parse(e.data);
    console.log(data.type, data.message);
};
```

Events include: `connected` (initial), `reminder` (note fired), and heartbeats every 30s.

## Conversations

- Use `POST /chat` with `{"message": "...", "session_id": "..."}` to interact with Jarvis.
- Omit `session_id` to start a new conversation (one will be returned).
- Use `POST /chat-with-context` to attach file/URL content as context.
- Responses may include `agent_steps` when multi-step reasoning was used.

## Scripts

Python scripts placed in the `scripts/` folder are automatically exposed as API endpoints under `/api/scripts/<name>`. Scripts are discovered on server startup. Use `POST /api/scripts/reload` to re-scan without restarting.

Scripts can define metadata via comments:
```python
# DESCRIPTION: Takes a screenshot and saves it
# ARGS: {"width": "int, optional", "output": "string, optional"}
```

Arguments are passed as `SCRIPT_*` environment variables (e.g., `{"width": "1920"}` becomes `SCRIPT_WIDTH=1920`).

## Mental Notes & Scheduled Commands

Notes can have:
- `due_date`: When to fire the reminder
- `command`: A shell command to execute when due
- `recurrence`: Pattern like `every 1m`, `every 2h`, `daily`, `weekly`
- `expires_at`: Auto-clear the note after this datetime

## Inference Parameters

Adjust LLM behavior in real-time via `PUT /api/inference`:
- `temperature` (0-2): Creativity level
- `top_p` (0-1): Nucleus sampling
- `max_tokens` (128-4096): Response length
- `frequency_penalty` / `presence_penalty` (-2 to 2): Repetition control

## LLM Provider Failover

Multiple API keys can be configured per provider. The system tries each in order:
`HUGGINGFACE_API_KEY`, `HUGGINGFACE_API_KEY_2`, `HUGGINGFACE_API_KEY_3`, etc.
Failover order is controlled by `LLM_FAILOVER_ORDER` (e.g., `huggingface,groq,gemini`).

## Rate Limits

Depends on your LLM provider's free tier. The system automatically switches to the next provider/key on 402 or 429 errors.

## Device Permissions & Location

Connected devices (phones, tablets) can grant browser permissions to share data with Jarvis:

- **GPS Location** — Device reports coordinates every 30s via `POST /api/device/location`. Jarvis can query all device positions via `GET /api/device/locations`.
- **Browser Notifications** — Push notifications for reminders when the tab is in the background.
- **Camera** — Reserved for future features (QR scanning, visual input).
- **Microphone** — Used for voice input (speech-to-text).

All permissions are OFF by default and toggled per-device in the Settings page. Preferences are stored in the browser's localStorage.

## Bluetooth & IMU Sensors

Jarvis can connect to Bluetooth Low Energy (BLE) devices like IMU sensors. The system provides:

- **Device scanning** — `POST /api/bluetooth/scan` discovers nearby BLE/classic devices
- **Sensor data ingestion** — Scripts reading IMU data call `POST /api/bluetooth/sensor-data` to feed accelerometer/gyroscope readings into Jarvis
- **Live data viewing** — The Bluetooth UI page shows real-time sensor data tables with auto-refresh
- **Fused context** — `GET /api/bluetooth/context` combines IMU data with phone GPS for room-level positioning

**Use case:** An IMU sensor on the user's body sends motion data via BLE to the Pi. A phone in the user's pocket reports GPS. A script reads both via `/api/bluetooth/context` to determine: "User is walking (IMU) in the living room (GPS)."

**Dependencies:** `pip install bleak` for BLE support. Falls back to system commands (bluetoothctl/PowerShell) without it.

## Database Management

Direct CRUD access to the SQLite database via REST API. Useful for creating notes, tasks, and reminders without prompting the LLM.

**Tables:** `notes`, `conversations`, `metrics`, `session_metadata`

**Notes table columns:**
- `content` (text): The note content
- `category` (text, optional): Category tag
- `due_date` (datetime, optional): When to fire the reminder (ISO format: `2026-05-06 14:30:00`)
- `command` (text, optional): Shell command to execute when due
- `recurrence` (text, optional): `every 1m`, `every 2h`, `daily`, `weekly`
- `expires_at` (datetime, optional): Auto-clear after this time
- `status`: `active` or `completed`

**Example — Create a recurring note via API:**
```json
POST /api/db/notes
{
  "content": "Check server health",
  "due_date": "2026-05-06 15:00:00",
  "command": "python health_check.py",
  "recurrence": "every 1h",
  "status": "active"
}
```
""",
        "version": "1.0.0",
    },
    "servers": [{"url": "/", "description": "Local server"}],
    "components": {
        "securitySchemes": {
            "basicAuth": {
                "type": "http",
                "scheme": "basic",
                "description": "HTTP Basic Auth using IAM credentials (username:password)"
            }
        }
    },
    "security": [{"basicAuth": []}],
    "paths": {
        "/chat": {
            "post": {
                "summary": "Send a message to Jarvis",
                "tags": ["Chat"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["message"],
                                "properties": {
                                    "message": {"type": "string", "description": "The user's message"},
                                    "session_id": {"type": "string", "description": "Optional session ID to continue a conversation"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Assistant response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "response": {"type": "string"},
                                        "session_id": {"type": "string"},
                                        "agent_steps": {"type": "array", "items": {"type": "object"}},
                                    },
                                }
                            }
                        },
                    },
                    "400": {"description": "Missing message field"},
                },
            }
        },
        "/health": {
            "get": {
                "summary": "Health check",
                "tags": ["System"],
                "responses": {
                    "200": {
                        "description": "Server status",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "provider": {"type": "string"},
                                        "capabilities": {"type": "array", "items": {"type": "string"}},
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/notifications": {
            "get": {
                "summary": "Get pending notifications (polling)",
                "tags": ["Notifications"],
                "responses": {
                    "200": {
                        "description": "List of pending notifications",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "message": {"type": "string"},
                                            "timestamp": {"type": "string"},
                                            "note_id": {"type": "integer"},
                                            "type": {"type": "string"},
                                        },
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/subscribe": {
            "get": {
                "summary": "Subscribe to real-time notifications (Server-Sent Events)",
                "tags": ["Notifications"],
                "description": "Opens an SSE stream. Clients receive events when Jarvis sends notifications or reminders.",
                "responses": {
                    "200": {
                        "description": "SSE event stream",
                        "content": {"text/event-stream": {"schema": {"type": "string"}}},
                    }
                },
            }
        },
        "/api/sessions": {
            "get": {
                "summary": "List past conversation sessions",
                "tags": ["Sessions"],
                "responses": {
                    "200": {
                        "description": "List of sessions",
                        "content": {"application/json": {"schema": {"type": "array", "items": {"type": "object"}}}},
                    }
                },
            }
        },
        "/api/sessions/{session_id}": {
            "get": {
                "summary": "Get full message history for a session",
                "tags": ["Sessions"],
                "parameters": [{"name": "session_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {
                    "200": {"description": "Message history", "content": {"application/json": {"schema": {"type": "array", "items": {"type": "object"}}}}},
                },
            }
        },
        "/api/sessions/{session_id}/export": {
            "get": {
                "summary": "Export conversation as XLSX",
                "tags": ["Sessions"],
                "parameters": [{"name": "session_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "XLSX file download"}},
            }
        },
        "/api/config": {
            "get": {
                "summary": "Get current configuration",
                "tags": ["Configuration"],
                "responses": {"200": {"description": "Configuration values"}},
            },
            "put": {
                "summary": "Update a configuration value",
                "tags": ["Configuration"],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string"},
                                    "value": {},
                                },
                            }
                        }
                    }
                },
                "responses": {"200": {"description": "Update result"}, "400": {"description": "Invalid value"}},
            },
        },
        "/api/inference": {
            "get": {
                "summary": "Get LLM inference parameters",
                "tags": ["Configuration"],
                "responses": {"200": {"description": "Current inference params"}},
            },
            "put": {
                "summary": "Update LLM inference parameters",
                "tags": ["Configuration"],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                                    "top_p": {"type": "number", "minimum": 0, "maximum": 1},
                                    "max_tokens": {"type": "integer", "minimum": 1, "maximum": 4096},
                                    "frequency_penalty": {"type": "number", "minimum": -2, "maximum": 2},
                                    "presence_penalty": {"type": "number", "minimum": -2, "maximum": 2},
                                },
                            }
                        }
                    }
                },
                "responses": {"200": {"description": "Updated params"}},
            },
        },
        "/api/metrics": {
            "get": {
                "summary": "Get usage metrics summary",
                "tags": ["Monitoring"],
                "responses": {"200": {"description": "Metrics summary"}},
            }
        },
        "/api/metrics/hourly": {
            "get": {
                "summary": "Get hourly metrics breakdown (last 24h)",
                "tags": ["Monitoring"],
                "responses": {"200": {"description": "Hourly breakdown"}},
            }
        },
        "/api/system": {
            "get": {
                "summary": "Get system resource usage (CPU, RAM, disk)",
                "tags": ["Monitoring"],
                "responses": {"200": {"description": "System metrics"}},
            }
        },
        "/api/search": {
            "get": {
                "summary": "Search across past conversations",
                "tags": ["Sessions"],
                "parameters": [{"name": "q", "in": "query", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "Search results"}},
            }
        },
        "/api/upload": {
            "post": {
                "summary": "Upload a file for context extraction",
                "tags": ["Context"],
                "requestBody": {"content": {"multipart/form-data": {"schema": {"type": "object", "properties": {"file": {"type": "string", "format": "binary"}}}}}},
                "responses": {"200": {"description": "Extracted content"}},
            }
        },
        "/api/fetch-url": {
            "post": {
                "summary": "Fetch URL content for context",
                "tags": ["Context"],
                "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"url": {"type": "string"}}}}}},
                "responses": {"200": {"description": "Extracted content"}},
            }
        },
        "/api/plugins": {
            "get": {
                "summary": "List installed plugins",
                "tags": ["Plugins"],
                "responses": {"200": {"description": "Plugin list"}},
            },
            "post": {
                "summary": "Install a new plugin",
                "tags": ["Plugins"],
                "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"name": {"type": "string"}, "code": {"type": "string"}}}}}},
                "responses": {"200": {"description": "Installation result"}},
            },
        },
        "/api/restart/status": {
            "get": {
                "summary": "Check if a server restart is pending",
                "tags": ["System"],
                "description": "Returns whether a restart is imminent and how many seconds remain. Used by the frontend to show a countdown snackbar.",
                "responses": {"200": {"description": "Restart status", "content": {"application/json": {"schema": {"type": "object", "properties": {"pending": {"type": "boolean"}, "remaining_seconds": {"type": "number"}}}}}}},
            }
        },
        "/api/restart": {
            "post": {
                "summary": "Manually trigger a server restart",
                "tags": ["System"],
                "description": "Initiates a server restart with a 3-second delay. The frontend will show a red snackbar countdown.",
                "responses": {"200": {"description": "Restart initiated"}},
            }
        },
        "/api/device/location": {
            "post": {
                "summary": "Report device GPS location",
                "tags": ["Devices"],
                "description": "Called by client devices to report their GPS coordinates. Jarvis stores the latest position per device.",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"type": "object", "required": ["latitude", "longitude"], "properties": {
                        "latitude": {"type": "number", "description": "GPS latitude"},
                        "longitude": {"type": "number", "description": "GPS longitude"},
                        "accuracy": {"type": "number", "description": "Accuracy in meters"},
                        "device_id": {"type": "string", "description": "Unique device identifier (defaults to IP)"},
                        "timestamp": {"type": "string", "description": "ISO timestamp of the reading"},
                    }}}},
                },
                "responses": {"200": {"description": "Location stored"}},
            }
        },
        "/api/device/locations": {
            "get": {
                "summary": "Get all known device locations",
                "tags": ["Devices"],
                "description": "Returns the last known GPS position of all devices that have reported their location.",
                "responses": {"200": {"description": "List of device locations"}},
            }
        },
        "/api/bluetooth/scan": {
            "post": {
                "summary": "Scan for nearby Bluetooth devices",
                "tags": ["Bluetooth"],
                "description": "Initiates a BLE/classic Bluetooth scan. Returns discovered devices.",
                "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"duration": {"type": "integer", "default": 5, "description": "Scan duration in seconds"}}}}}},
                "responses": {"200": {"description": "List of discovered devices"}},
            }
        },
        "/api/bluetooth/devices": {
            "get": {
                "summary": "List all known Bluetooth devices",
                "tags": ["Bluetooth"],
                "responses": {"200": {"description": "Device list with connection status and sensor data availability"}},
            }
        },
        "/api/bluetooth/sensor-data": {
            "post": {
                "summary": "Report IMU sensor data from a Bluetooth device",
                "tags": ["Bluetooth"],
                "description": "Called by scripts reading BLE IMU sensors. Stores accelerometer/gyroscope data for the device.",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"type": "object", "required": ["address"], "properties": {
                        "address": {"type": "string", "description": "BLE device MAC address"},
                        "accel_x": {"type": "number"}, "accel_y": {"type": "number"}, "accel_z": {"type": "number"},
                        "gyro_x": {"type": "number"}, "gyro_y": {"type": "number"}, "gyro_z": {"type": "number"},
                        "mag_x": {"type": "number"}, "mag_y": {"type": "number"}, "mag_z": {"type": "number"},
                    }}}},
                },
                "responses": {"200": {"description": "Data stored"}},
            }
        },
        "/api/bluetooth/sensor-data/{address}": {
            "get": {
                "summary": "Get recent sensor readings for a device",
                "tags": ["Bluetooth"],
                "parameters": [
                    {"name": "address", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "last", "in": "query", "schema": {"type": "integer", "default": 10}},
                ],
                "responses": {"200": {"description": "Recent sensor readings (accel, gyro, mag)"}},
            }
        },
        "/api/bluetooth/context": {
            "get": {
                "summary": "Get fused sensor + location context",
                "tags": ["Bluetooth"],
                "description": "Combines IMU sensor data with phone GPS locations. Scripts can poll this to know both motion state and room-level position.",
                "responses": {"200": {"description": "Fused context with IMU data and device locations"}},
            }
        },
        "/api/iam/users": {
            "get": {
                "summary": "List all users (admin only)",
                "tags": ["IAM"],
                "responses": {"200": {"description": "User list"}},
            },
            "post": {
                "summary": "Create a new user (admin only)",
                "tags": ["IAM"],
                "requestBody": {"content": {"application/json": {"schema": {"type": "object", "required": ["username", "password"], "properties": {
                    "username": {"type": "string"},
                    "password": {"type": "string"},
                    "role": {"type": "string", "default": "user", "description": "Role name: admin, user, viewer, or custom"},
                }}}}},
                "responses": {"200": {"description": "User created"}, "400": {"description": "User exists or invalid role"}},
            },
        },
        "/api/iam/users/{username}": {
            "put": {
                "summary": "Update a user (admin only)",
                "tags": ["IAM"],
                "parameters": [{"name": "username", "in": "path", "required": True, "schema": {"type": "string"}}],
                "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {
                    "role": {"type": "string"}, "active": {"type": "boolean"}, "password": {"type": "string"},
                }}}}},
                "responses": {"200": {"description": "User updated"}},
            },
            "delete": {
                "summary": "Delete a user (admin only)",
                "tags": ["IAM"],
                "parameters": [{"name": "username", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "User deleted"}},
            },
        },
        "/api/iam/roles": {
            "get": {
                "summary": "List all roles with permissions (admin only)",
                "tags": ["IAM"],
                "responses": {"200": {"description": "Role list with permission arrays"}},
            },
            "post": {
                "summary": "Create or update a role (admin only)",
                "tags": ["IAM"],
                "requestBody": {"content": {"application/json": {"schema": {"type": "object", "required": ["name", "permissions"], "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "permissions": {"type": "array", "items": {"type": "string"}, "description": "Permission group names: chat, sessions, notes, commands, config, monitoring, database, bluetooth, files, plugins, devices, admin"},
                }}}}},
                "responses": {"200": {"description": "Role updated"}},
            },
        },
        "/api/iam/permissions": {
            "get": {
                "summary": "List all available permission groups (admin only)",
                "tags": ["IAM"],
                "description": "Returns all permission groups with descriptions and associated endpoints.",
                "responses": {"200": {"description": "Permission groups"}},
            }
        },
        "/api/db/tables": {
            "get": {
                "summary": "List all database tables with row counts",
                "tags": ["Database"],
                "responses": {"200": {"description": "List of tables", "content": {"application/json": {"schema": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "row_count": {"type": "integer"}}}}}}}},
            }
        },
        "/api/db/{table_name}": {
            "get": {
                "summary": "Get rows from a table",
                "tags": ["Database"],
                "parameters": [
                    {"name": "table_name", "in": "path", "required": True, "schema": {"type": "string"}, "description": "Table name (notes, conversations, metrics, session_metadata)"},
                    {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 100}},
                    {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
                ],
                "responses": {"200": {"description": "Table data with columns and rows"}},
            },
            "post": {
                "summary": "Insert a new row into a table",
                "tags": ["Database"],
                "parameters": [{"name": "table_name", "in": "path", "required": True, "schema": {"type": "string"}}],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"type": "object", "description": "Column-value pairs. Example for notes: {content, due_date, command, recurrence, status, expires_at}"}}},
                },
                "responses": {"200": {"description": "Insert result with new row ID"}, "400": {"description": "Invalid data"}},
            },
        },
        "/api/db/{table_name}/{row_id}": {
            "put": {
                "summary": "Update a row by ID",
                "tags": ["Database"],
                "parameters": [
                    {"name": "table_name", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "row_id", "in": "path", "required": True, "schema": {"type": "integer"}},
                ],
                "requestBody": {"content": {"application/json": {"schema": {"type": "object", "description": "Fields to update"}}}},
                "responses": {"200": {"description": "Update result"}, "404": {"description": "Row not found"}},
            },
            "delete": {
                "summary": "Delete a row by ID",
                "tags": ["Database"],
                "parameters": [
                    {"name": "table_name", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "row_id", "in": "path", "required": True, "schema": {"type": "integer"}},
                ],
                "responses": {"200": {"description": "Delete result"}, "404": {"description": "Row not found"}},
            },
        },
        "/api/briefing": {
            "get": {
                "summary": "Generate daily briefing for the current user",
                "tags": ["Briefing"],
                "description": "Combines weather, calendar, notes, and metrics into a morning summary.",
                "responses": {"200": {"description": "Briefing text", "content": {"application/json": {"schema": {"type": "object", "properties": {"briefing": {"type": "string"}}}}}}},
            }
        },
        "/api/briefing/settings": {
            "get": {
                "summary": "Get briefing settings for the current user",
                "tags": ["Briefing"],
                "responses": {"200": {"description": "Briefing settings"}},
            },
            "put": {
                "summary": "Update briefing settings",
                "tags": ["Briefing"],
                "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {
                    "enabled": {"type": "boolean"},
                    "time": {"type": "string", "description": "Delivery time in HH:MM format"},
                    "include_weather": {"type": "boolean"},
                    "include_calendar": {"type": "boolean"},
                    "include_notes": {"type": "boolean"},
                    "include_metrics": {"type": "boolean"},
                    "include_quote": {"type": "boolean"},
                    "city": {"type": "string", "description": "Weather city override"},
                }}}}},
                "responses": {"200": {"description": "Settings saved"}},
            },
        },
        "/api/workflows": {
            "get": {
                "summary": "List all workflows",
                "tags": ["Workflows"],
                "description": "Returns all workflows for the current user (admins see all).",
                "responses": {"200": {"description": "Workflow list"}},
            },
            "post": {
                "summary": "Create a new workflow",
                "tags": ["Workflows"],
                "description": "Define a trigger→action automation rule.",
                "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["name", "trigger_type", "trigger_config", "action_type", "action_config"], "properties": {
                    "name": {"type": "string", "description": "Workflow name"},
                    "description": {"type": "string"},
                    "trigger_type": {"type": "string", "enum": ["schedule", "gps_enter", "gps_exit", "event"]},
                    "trigger_config": {"type": "object", "description": "Trigger configuration. Schedule: {time, days/interval}. GPS: {latitude, longitude, radius_meters}. Event: {event_name}."},
                    "action_type": {"type": "string", "enum": ["notify", "run_command", "briefing", "smart_home", "note", "webhook"]},
                    "action_config": {"type": "object", "description": "Action configuration. Notify: {message}. Command: {command, notify_output}. Smart home: {action, entity_id}. Note: {content}. Webhook: {url, method, payload}."},
                    "conditions": {"type": "object", "description": "Optional conditions: {time_range: {after, before}, day_of_week: [...]}"},
                }}}}},
                "responses": {"200": {"description": "Workflow created"}, "400": {"description": "Invalid data"}},
            },
        },
        "/api/workflows/{workflow_id}": {
            "get": {
                "summary": "Get a workflow by ID",
                "tags": ["Workflows"],
                "parameters": [{"name": "workflow_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "responses": {"200": {"description": "Workflow details"}, "404": {"description": "Not found"}},
            },
            "put": {
                "summary": "Update a workflow",
                "tags": ["Workflows"],
                "parameters": [{"name": "workflow_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {
                    "name": {"type": "string"}, "enabled": {"type": "boolean"},
                    "trigger_type": {"type": "string"}, "trigger_config": {"type": "object"},
                    "action_type": {"type": "string"}, "action_config": {"type": "object"},
                }}}}},
                "responses": {"200": {"description": "Updated"}, "400": {"description": "Invalid data"}},
            },
            "delete": {
                "summary": "Delete a workflow",
                "tags": ["Workflows"],
                "parameters": [{"name": "workflow_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "responses": {"200": {"description": "Deleted"}, "404": {"description": "Not found"}},
            },
        },
        "/api/workflows/{workflow_id}/logs": {
            "get": {
                "summary": "Get execution logs for a workflow",
                "tags": ["Workflows"],
                "parameters": [
                    {"name": "workflow_id", "in": "path", "required": True, "schema": {"type": "integer"}},
                    {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
                ],
                "responses": {"200": {"description": "Execution log entries"}},
            }
        },
        "/api/workflows/{workflow_id}/test": {
            "post": {
                "summary": "Manually trigger a workflow for testing",
                "tags": ["Workflows"],
                "parameters": [{"name": "workflow_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "responses": {"200": {"description": "Workflow triggered"}},
            }
        },
        "/api/sandbox/run": {
            "post": {
                "summary": "Execute Python code in the sandbox",
                "tags": ["Code Sandbox"],
                "description": "Runs Python code in a restricted environment with timeout. Safe modules: math, json, datetime, random, re, collections, itertools, functools.",
                "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["code"], "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "default": 10, "maximum": 30, "description": "Max execution time in seconds"},
                }}}}},
                "responses": {"200": {"description": "Execution result", "content": {"application/json": {"schema": {"type": "object", "properties": {
                    "output": {"type": "string", "description": "stdout output"},
                    "error": {"type": "string", "description": "stderr or exception"},
                    "result": {"type": "string", "description": "Last expression value"},
                }}}}}},
            }
        },
        "/api/suggestions": {
            "get": {
                "summary": "Get contextual suggestions for the current user",
                "tags": ["Suggestions"],
                "description": "Analyzes user activity patterns and returns proactive automation suggestions.",
                "responses": {"200": {"description": "List of suggestions with actions"}},
            }
        },
        "/api/suggestions/history": {
            "get": {
                "summary": "Get past suggestions history",
                "tags": ["Suggestions"],
                "parameters": [{"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}}],
                "responses": {"200": {"description": "Past suggestions with accept/dismiss status"}},
            }
        },
        "/api/suggestions/{suggestion_id}/dismiss": {
            "post": {
                "summary": "Dismiss a suggestion",
                "tags": ["Suggestions"],
                "parameters": [{"name": "suggestion_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "responses": {"200": {"description": "Dismissed"}},
            }
        },
        "/api/suggestions/{suggestion_id}/accept": {
            "post": {
                "summary": "Accept a suggestion",
                "tags": ["Suggestions"],
                "parameters": [{"name": "suggestion_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "responses": {"200": {"description": "Accepted"}},
            }
        },
        "/api/suggestions/activity": {
            "get": {
                "summary": "Get user activity statistics",
                "tags": ["Suggestions"],
                "description": "Returns activity breakdown, top commands, and peak usage hours for the last 7 days.",
                "responses": {"200": {"description": "Activity stats"}},
            }
        },
        "/api/suggestions/ingest-history": {
            "post": {
                "summary": "Ingest OS command history",
                "tags": ["Suggestions"],
                "description": "Reads the OS shell history (bash_history, PowerShell) and ingests new commands for pattern analysis.",
                "responses": {"200": {"description": "Ingestion result with count"}},
            }
        },
        "/api/cron": {
            "get": {
                "summary": "List all cron jobs",
                "tags": ["Cron Jobs"],
                "responses": {"200": {"description": "List of scheduled jobs with status and history"}},
            },
            "post": {
                "summary": "Create a new cron job",
                "tags": ["Cron Jobs"],
                "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["name", "command", "schedule"], "properties": {
                    "name": {"type": "string", "description": "Job name"},
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "schedule": {"type": "string", "description": "Schedule: 'every 5m', 'daily 08:00', 'hourly', '*/5 * * * *'"},
                    "description": {"type": "string"},
                }}}}},
                "responses": {"200": {"description": "Job created"}},
            },
        },
        "/api/cron/{job_id}": {
            "get": {"summary": "Get a cron job", "tags": ["Cron Jobs"], "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "integer"}}], "responses": {"200": {"description": "Job details"}}},
            "put": {"summary": "Update a cron job", "tags": ["Cron Jobs"], "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "integer"}}], "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"name": {"type": "string"}, "command": {"type": "string"}, "schedule": {"type": "string"}, "enabled": {"type": "boolean"}}}}}}, "responses": {"200": {"description": "Updated"}}},
            "delete": {"summary": "Delete a cron job", "tags": ["Cron Jobs"], "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "integer"}}], "responses": {"200": {"description": "Deleted"}}},
        },
        "/api/cron/{job_id}/run": {
            "post": {"summary": "Manually trigger a cron job", "tags": ["Cron Jobs"], "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "integer"}}], "responses": {"200": {"description": "Execution result with output"}}}
        },
        "/api/cron/{job_id}/history": {
            "get": {"summary": "Get execution history for a cron job", "tags": ["Cron Jobs"], "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "integer"}}, {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}}], "responses": {"200": {"description": "Execution history"}}}
        },
        "/api/logs/watches": {
            "get": {"summary": "List watched log files", "tags": ["Log Analyzer"], "responses": {"200": {"description": "List of watched files with thresholds"}}},
            "post": {"summary": "Add a log file to watch", "tags": ["Log Analyzer"], "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["file_path"], "properties": {"file_path": {"type": "string"}, "label": {"type": "string"}, "error_threshold": {"type": "integer", "default": 5}, "window_minutes": {"type": "integer", "default": 5}}}}}}, "responses": {"200": {"description": "Watch added"}}},
        },
        "/api/logs/tail": {
            "get": {"summary": "Tail a log file", "tags": ["Log Analyzer"], "parameters": [{"name": "file", "in": "query", "required": True, "schema": {"type": "string"}}, {"name": "lines", "in": "query", "schema": {"type": "integer", "default": 50}}], "responses": {"200": {"description": "Last N lines of the file"}}}
        },
        "/api/logs/search": {
            "get": {"summary": "Search a log file with regex", "tags": ["Log Analyzer"], "parameters": [{"name": "file", "in": "query", "required": True, "schema": {"type": "string"}}, {"name": "pattern", "in": "query", "required": True, "schema": {"type": "string"}}], "responses": {"200": {"description": "Matching lines"}}}
        },
        "/api/logs/errors": {
            "get": {"summary": "Get error summary for a watched file", "tags": ["Log Analyzer"], "parameters": [{"name": "file", "in": "query", "required": True, "schema": {"type": "string"}}], "responses": {"200": {"description": "Error count and recent errors"}}}
        },
        "/api/logs/alerts": {
            "get": {"summary": "Get log alerts", "tags": ["Log Analyzer"], "parameters": [{"name": "unacknowledged", "in": "query", "schema": {"type": "string", "enum": ["true", "false"]}}], "responses": {"200": {"description": "Alert list"}}}
        },
        "/api/logs/alerts/{alert_id}/acknowledge": {
            "post": {"summary": "Acknowledge a log alert", "tags": ["Log Analyzer"], "parameters": [{"name": "alert_id", "in": "path", "required": True, "schema": {"type": "integer"}}], "responses": {"200": {"description": "Acknowledged"}}}
        },
        "/api/services": {
            "get": {"summary": "List system services", "tags": ["Services"], "description": "Lists systemd services (Linux) or Windows services. Supports remote via ?host=name.", "parameters": [{"name": "host", "in": "query", "schema": {"type": "string", "description": "Remote SSH host (optional)"}}], "responses": {"200": {"description": "Service list"}}}
        },
        "/api/services/{service_name}/{action}": {
            "post": {"summary": "Control a system service", "tags": ["Services"], "description": "Actions: start, stop, restart, enable, disable. Admin only.", "parameters": [{"name": "service_name", "in": "path", "required": True, "schema": {"type": "string"}}, {"name": "action", "in": "path", "required": True, "schema": {"type": "string", "enum": ["start", "stop", "restart", "enable", "disable"]}}], "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"host": {"type": "string"}}}}}}, "responses": {"200": {"description": "Action result"}}}
        },
        "/api/services/{service_name}/status": {
            "get": {"summary": "Get service status", "tags": ["Services"], "parameters": [{"name": "service_name", "in": "path", "required": True, "schema": {"type": "string"}}, {"name": "host", "in": "query", "schema": {"type": "string"}}], "responses": {"200": {"description": "Service status"}}}
        },
        "/api/docker/containers": {
            "get": {"summary": "List Docker containers", "tags": ["Docker"], "parameters": [{"name": "all", "in": "query", "schema": {"type": "string", "enum": ["true", "false"]}}, {"name": "host", "in": "query", "schema": {"type": "string"}}], "responses": {"200": {"description": "Container list"}}}
        },
        "/api/docker/containers/{container}/{action}": {
            "post": {"summary": "Control a Docker container", "tags": ["Docker"], "description": "Actions: start, stop, restart, pause, unpause. Admin only.", "parameters": [{"name": "container", "in": "path", "required": True, "schema": {"type": "string"}}, {"name": "action", "in": "path", "required": True, "schema": {"type": "string", "enum": ["start", "stop", "restart", "pause", "unpause"]}}], "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"host": {"type": "string"}}}}}}, "responses": {"200": {"description": "Action result"}}}
        },
        "/api/docker/containers/{container}/logs": {
            "get": {"summary": "Get Docker container logs", "tags": ["Docker"], "parameters": [{"name": "container", "in": "path", "required": True, "schema": {"type": "string"}}, {"name": "lines", "in": "query", "schema": {"type": "integer", "default": 50}}, {"name": "host", "in": "query", "schema": {"type": "string"}}], "responses": {"200": {"description": "Container logs"}}}
        },
        "/api/docker/stats": {
            "get": {"summary": "Get Docker resource stats", "tags": ["Docker"], "parameters": [{"name": "host", "in": "query", "schema": {"type": "string"}}], "responses": {"200": {"description": "CPU, memory, network per container"}}}
        },
        "/api/git/status": {
            "get": {"summary": "Get git repository status", "tags": ["Git"], "parameters": [{"name": "repo", "in": "query", "schema": {"type": "string", "default": "."}}, {"name": "host", "in": "query", "schema": {"type": "string"}}], "responses": {"200": {"description": "Branch, changes, modified files"}}}
        },
        "/api/git/pull": {
            "post": {"summary": "Pull latest changes", "tags": ["Git"], "description": "Admin only.", "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"repo": {"type": "string", "default": "."}, "host": {"type": "string"}}}}}}, "responses": {"200": {"description": "Pull result"}}}
        },
        "/api/git/log": {
            "get": {"summary": "Get recent git commits", "tags": ["Git"], "parameters": [{"name": "repo", "in": "query", "schema": {"type": "string", "default": "."}}, {"name": "count", "in": "query", "schema": {"type": "integer", "default": 10}}, {"name": "host", "in": "query", "schema": {"type": "string"}}], "responses": {"200": {"description": "Commit list"}}}
        },
        "/api/backups": {
            "get": {"summary": "List backup history", "tags": ["Backups"], "description": "Admin only. Shows past backups with size, duration, and status.", "responses": {"200": {"description": "Backup history"}}}
        },
        "/api/backups/run": {
            "post": {"summary": "Run a backup now", "tags": ["Backups"], "description": "Admin only. Creates a ZIP archive of DB, configs, scripts, plugins.", "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"type": {"type": "string", "enum": ["full", "db", "config"], "default": "full"}}}}}}, "responses": {"200": {"description": "Backup result with path, size, checksum"}}}
        },
        "/api/backups/config": {
            "get": {"summary": "Get backup configuration", "tags": ["Backups"], "responses": {"200": {"description": "Backup settings"}}},
            "put": {"summary": "Update backup configuration", "tags": ["Backups"], "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"enabled": {"type": "string"}, "frequency": {"type": "string", "enum": ["hourly", "daily", "weekly"]}, "time": {"type": "string"}, "retention_days": {"type": "string"}, "remote_enabled": {"type": "string"}, "remote_host": {"type": "string"}, "remote_path": {"type": "string"}, "remote_user": {"type": "string"}, "include_uploads": {"type": "string"}}}}}}, "responses": {"200": {"description": "Config saved"}}},
        },
        "/api/backups/{backup_id}/verify": {
            "post": {"summary": "Verify backup integrity", "tags": ["Backups"], "description": "Checks SHA256 checksum of a backup file.", "parameters": [{"name": "backup_id", "in": "path", "required": True, "schema": {"type": "integer"}}], "responses": {"200": {"description": "Verification result"}}}
        },
        "/api/backups/heartbeat": {
            "get": {"summary": "Check if Jarvis is alive (public)", "tags": ["Backups"], "description": "No auth required. Returns health status based on last heartbeat. Use for external monitoring.", "responses": {"200": {"description": "Heartbeat status: healthy/warning/critical"}}}
        },
        "/api/backups/dr-plan": {
            "get": {"summary": "Generate disaster recovery plan", "tags": ["Backups"], "description": "Admin only. Returns step-by-step recovery instructions, RPO, and recommendations.", "responses": {"200": {"description": "DR plan"}}}
        },
        "/api/wifi/devices": {
            "get": {"summary": "List known WiFi network devices", "tags": ["WiFi"], "responses": {"200": {"description": "Device list with IP, MAC, hostname, SSH status"}}}
        },
        "/api/wifi/scan": {
            "post": {"summary": "Scan the network for devices", "tags": ["WiFi"], "description": "Discovers devices via ARP table and updates the database.", "responses": {"200": {"description": "Scan results"}}}
        },
        "/api/wifi/devices/{mac}": {
            "put": {"summary": "Update device metadata", "tags": ["WiFi"], "parameters": [{"name": "mac", "in": "path", "required": True, "schema": {"type": "string"}}], "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"custom_name": {"type": "string"}, "ssh_username": {"type": "string"}, "ssh_password": {"type": "string"}, "ssh_port": {"type": "integer"}, "notes": {"type": "string"}}}}}}, "responses": {"200": {"description": "Updated"}}}
        },
        "/api/wifi/ssh/connect": {
            "post": {"summary": "SSH connect to a device", "tags": ["WiFi"], "description": "Attempts SSH connection. Auto-tries default credentials if auto_try is true.", "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["ip"], "properties": {"ip": {"type": "string"}, "username": {"type": "string"}, "password": {"type": "string"}, "port": {"type": "integer", "default": 22}, "auto_try": {"type": "boolean", "default": True}}}}}}, "responses": {"200": {"description": "Connected"}, "401": {"description": "Auth failed, needs credentials"}}}
        },
        "/api/wifi/ssh/disconnect": {
            "post": {"summary": "Disconnect SSH session", "tags": ["WiFi"], "requestBody": {"content": {"application/json": {"schema": {"type": "object", "required": ["ip"], "properties": {"ip": {"type": "string"}}}}}}, "responses": {"200": {"description": "Disconnected"}}}
        },
        "/api/wifi/ssh/execute": {
            "post": {"summary": "Execute command on connected device", "tags": ["WiFi"], "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["ip", "command"], "properties": {"ip": {"type": "string"}, "command": {"type": "string"}, "timeout": {"type": "integer", "default": 30}}}}}}, "responses": {"200": {"description": "Command output"}}}
        },
        "/api/wifi/sessions": {
            "get": {"summary": "List active SSH sessions", "tags": ["WiFi"], "responses": {"200": {"description": "Active sessions with IP, username, connected_at"}}}
        },
        "/api/assistants": {
            "get": {"summary": "List custom assistants", "tags": ["Assistants"], "description": "Returns assistants owned by the user plus shared ones.", "responses": {"200": {"description": "Assistant list"}}},
            "post": {"summary": "Create a custom assistant", "tags": ["Assistants"], "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["name", "system_prompt"], "properties": {"name": {"type": "string"}, "description": {"type": "string"}, "system_prompt": {"type": "string", "description": "Full system prompt text"}, "icon": {"type": "string", "default": "🤖"}, "shared": {"type": "boolean", "default": False}, "inference_params": {"type": "object", "properties": {"temperature": {"type": "number"}, "max_tokens": {"type": "integer"}, "top_p": {"type": "number"}, "frequency_penalty": {"type": "number"}, "presence_penalty": {"type": "number"}}}}}}}}, "responses": {"200": {"description": "Assistant created"}}},
        },
        "/api/assistants/{assistant_id}": {
            "get": {"summary": "Get assistant details", "tags": ["Assistants"], "parameters": [{"name": "assistant_id", "in": "path", "required": True, "schema": {"type": "integer"}}], "responses": {"200": {"description": "Full assistant config"}}},
            "put": {"summary": "Update an assistant (owner or admin)", "tags": ["Assistants"], "parameters": [{"name": "assistant_id", "in": "path", "required": True, "schema": {"type": "integer"}}], "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}, "system_prompt": {"type": "string"}, "icon": {"type": "string"}, "shared": {"type": "boolean"}, "inference_params": {"type": "object"}}}}}}, "responses": {"200": {"description": "Updated"}}},
            "delete": {"summary": "Delete an assistant (owner or admin)", "tags": ["Assistants"], "parameters": [{"name": "assistant_id", "in": "path", "required": True, "schema": {"type": "integer"}}], "responses": {"200": {"description": "Deleted"}}},
        },
        "/api/assistants/default-prompt": {
            "get": {"summary": "Get the default Jarvis system prompt", "tags": ["Assistants"], "description": "Returns the built-in Jarvis system prompt as a template for creating custom assistants.", "responses": {"200": {"description": "Default prompt text"}}}
        },
        "/api/notes/{note_id}/snooze": {
            "post": {"summary": "Snooze a reminder", "tags": ["Notifications"], "description": "Postpones a fired reminder by N minutes. Reschedules the note's due_date.", "parameters": [{"name": "note_id", "in": "path", "required": True, "schema": {"type": "integer"}}], "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["minutes"], "properties": {"minutes": {"type": "integer", "minimum": 1, "maximum": 1440}}}}}}, "responses": {"200": {"description": "Snoozed with new due time"}}}
        },
    },
}


SWAGGER_UI_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>J.A.R.V.I.S. API Documentation</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
        SwaggerUIBundle({
            url: '/api/spec',
            dom_id: '#swagger-ui',
            deepLinking: true,
            presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
            layout: "BaseLayout"
        });
    </script>
</body>
</html>"""
