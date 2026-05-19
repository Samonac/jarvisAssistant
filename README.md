# J.A.R.V.I.S. Assistant

A lightweight Flask-based personal assistant designed to run on a Raspberry Pi Zero 2W. Powered by free-tier LLM APIs with the personality of Jarvis from Iron Man.

## Features

- **Conversational AI** — Natural language interaction via free-tier LLMs (Groq, HuggingFace, Gemini)
- **Jarvis Personality** — Formal, witty, British-accented responses
- **System Commands** — Execute Linux bash commands with safety blocklist
- **Web Search** — Search the web via DuckDuckGo
- **Network Scanning** — Discover devices on your local network (ARP scan, ip neigh, /proc/net/arp)
- **Calendar Integration** — Google Calendar or CalDAV support
- **Email Sending** — Send emails via SMTP with confirmation
- **Mental Notes** — Persistent memory for reminders and to-do items
- **Web Frontend** — Browser-based chat, dashboard, and settings (HTMX + Alpine.js)
- **KPI Dashboard** — Track LLM calls, response times, tool usage, error rates
- **System Monitoring** — Real-time CPU, RAM, and disk usage from /proc

## Target Hardware

| Spec | Value |
|------|-------|
| Board | Raspberry Pi Zero 2W |
| CPU | ARM Cortex-A53 quad-core @ 1 GHz |
| RAM | 512 MB (shared with GPU) |
| Storage | microSD card (8 GB minimum, 16 GB+ recommended) |
| Network | WiFi 802.11 b/g/n (2.4 GHz) |
| OS | Raspberry Pi OS Lite (Bookworm, 64-bit recommended) |

## Prerequisites

- Raspberry Pi Zero 2W with Raspberry Pi OS Lite flashed and WiFi configured
- SSH access enabled (or keyboard + monitor)
- Python 3.10+ (comes pre-installed on Raspberry Pi OS Bookworm)
- Internet access (for LLM API calls and pip installs)
- A free-tier API key from one of: **Groq**, **HuggingFace**, or **Google Gemini**

## Installation (Step-by-Step for Raspberry Pi)

### 1. Update the system and install dependencies

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv python3-dev \
    libffi-dev libssl-dev arp-scan git
```

### 2. Download the HTMX and Alpine.js libraries

The project includes stub JS files that need to be replaced with the real libraries:

```bash
cd /home/pi
git clone <your-repo-url> JarvisAssistant
cd JarvisAssistant

# Download real HTMX (~14KB gzipped)
curl -sL https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js -o app/static/js/htmx.min.js

# Download real Alpine.js (~8KB gzipped)
curl -sL https://unpkg.com/alpinejs@3.14.3/dist/cdn.min.js -o app/static/js/alpine.min.js
```

### 3. Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** On the Pi Zero 2W, `pip install` can take several minutes for packages with C extensions. Be patient.

### 5. Configure environment variables

```bash
cp .env.example .env
nano .env
```

**Minimum required settings:**

```env
LLM_API_KEY=your-api-key-here
LLM_PROVIDER=groq
WEB_USERNAME=admin
WEB_PASSWORD=your-secure-password
```

See the full [Configuration Reference](#configuration-reference) below for all options.

### 6. Set up network scanning permissions

```bash
# Allow the pi user to run arp-scan without a password prompt
sudo bash -c 'echo "pi ALL=(ALL) NOPASSWD: /usr/sbin/arp-scan" >> /etc/sudoers.d/jarvis'
sudo chmod 440 /etc/sudoers.d/jarvis
```

### 7. Run the server

```bash
source venv/bin/activate
python run.py
```

The server starts on `http://0.0.0.0:5000`. Access it from any device on your local network at:

```
http://<your-pi-ip>:5000
```

To find your Pi's IP address:

```bash
hostname -I
```

### 8. (Optional) Reduce GPU memory to free RAM

Since this is a headless server, you can reclaim GPU memory:

```bash
sudo bash -c 'echo "gpu_mem=16" >> /boot/config.txt'
sudo reboot
```

This frees ~100 MB of RAM for the application.

## Quick Start (TL;DR)

```bash
# On a fresh Raspberry Pi OS Lite:
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv python3-dev libffi-dev libssl-dev arp-scan git
cd /home/pi && git clone <your-repo-url> JarvisAssistant && cd JarvisAssistant
curl -sL https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js -o app/static/js/htmx.min.js
curl -sL https://unpkg.com/alpinejs@3.14.3/dist/cdn.min.js -o app/static/js/alpine.min.js
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
cp .env.example .env && nano .env  # Set LLM_API_KEY, LLM_PROVIDER, WEB_USERNAME, WEB_PASSWORD
python run.py
```

## Configuration Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_API_KEY` | Yes | — | API key for the LLM provider |
| `LLM_PROVIDER` | Yes | — | `groq`, `huggingface`, or `gemini` |
| `PORT` | No | `5000` | Server port |
| `WEB_USERNAME` | No | `admin` | Web frontend username |
| `WEB_PASSWORD` | No | — | Web frontend password (required for auth) |
| `SECRET_KEY` | No | auto-generated | Flask session secret |
| `DATABASE_PATH` | No | `jarvis.db` | SQLite database file path |
| `COMMAND_TIMEOUT` | No | `60` | Command execution timeout (seconds) |
| `SCAN_TIMEOUT` | No | `120` | Network scan timeout (seconds) |
| `MAX_HISTORY_PAIRS` | No | `10` | Conversation history limit |
| `COMMAND_BLOCKLIST` | No | Linux defaults | Comma-separated dangerous command patterns |
| `CALENDAR_PROVIDER` | No | — | `google` or `caldav` |
| `GOOGLE_CREDENTIALS_PATH` | No | — | Path to Google service account JSON |
| `CALDAV_URL` | No | — | CalDAV server URL |
| `CALDAV_USERNAME` | No | — | CalDAV username |
| `CALDAV_PASSWORD` | No | — | CalDAV password |
| `SMTP_HOST` | No | — | SMTP server hostname |
| `SMTP_PORT` | No | `587` | SMTP server port |
| `SMTP_USERNAME` | No | — | SMTP login username |
| `SMTP_PASSWORD` | No | — | SMTP login password |
| `SMTP_FROM_ADDRESS` | No | — | Sender email address |
| `RETENTION_DAYS` | No | `30` | Days to keep conversation history |
| `REMINDER_WINDOW_MINUTES` | No | `15` | Minutes ahead for reminders |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send a message to Jarvis |
| GET | `/health` | Health check |
| GET | `/api/config` | Get current configuration |
| PUT | `/api/config` | Update a configuration value |
| GET | `/api/metrics` | Get usage metrics |
| GET | `/api/system` | Get system resource usage |

## Web Frontend

Access the web UI at `http://<pi-ip>:5000/`:
- **Chat** — Real-time conversation with Jarvis
- **Dashboard** — KPI metrics and system resource gauges
- **Settings** — View and modify configuration

## Running as a Service (Auto-Start on Boot)

Create a systemd service so Jarvis starts automatically when the Pi boots:

```bash
sudo nano /etc/systemd/system/jarvis.service
```

Paste the following:

```ini
[Unit]
Description=Jarvis Assistant
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/JarvisAssistant
Environment="PATH=/home/pi/JarvisAssistant/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
EnvironmentFile=/home/pi/JarvisAssistant/.env
ExecStart=/home/pi/JarvisAssistant/venv/bin/python run.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable jarvis
sudo systemctl start jarvis
```

Check status and logs:

```bash
sudo systemctl status jarvis
sudo journalctl -u jarvis -f
```

## Network Scanning

For full network scanning capabilities, the user running the server needs sudo access to `arp-scan`:

```bash
sudo bash -c 'echo "pi ALL=(ALL) NOPASSWD: /usr/sbin/arp-scan" >> /etc/sudoers.d/jarvis'
sudo chmod 440 /etc/sudoers.d/jarvis
```

Without sudo, the scanner falls back to `ip neigh` and `/proc/net/arp` (shows only previously-seen devices).

## Obtaining Free-Tier API Keys

### Groq (Recommended — fastest)
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys → Create API Key
4. Set `LLM_PROVIDER=groq` and paste the key as `LLM_API_KEY`

### HuggingFace
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" access
3. Set `LLM_PROVIDER=huggingface` and paste the token as `LLM_API_KEY`

### Google Gemini
1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Create an API key
3. Set `LLM_PROVIDER=gemini` and paste the key as `LLM_API_KEY`

## Setting Up Email (Optional)

For Gmail:
1. Enable 2-Factor Authentication on your Google account
2. Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
3. Generate an App Password for "Mail"
4. Configure in `.env`:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM_ADDRESS=your-email@gmail.com
```

## Setting Up Calendar (Optional)

### Google Calendar
1. Create a service account in [Google Cloud Console](https://console.cloud.google.com)
2. Enable the Google Calendar API
3. Download the credentials JSON file
4. Share your calendar with the service account email
5. Configure in `.env`:

```env
CALENDAR_PROVIDER=google
GOOGLE_CREDENTIALS_PATH=/home/pi/JarvisAssistant/credentials.json
```

### CalDAV (Nextcloud, Radicale, etc.)
```env
CALENDAR_PROVIDER=caldav
CALDAV_URL=https://your-server.com/remote.php/dav
CALDAV_USERNAME=your-username
CALDAV_PASSWORD=your-password
```

## Performance Notes

The Pi Zero 2W has limited resources. Here's what to expect:

| Metric | Typical Value |
|--------|---------------|
| Idle RAM usage | ~80-100 MB |
| Active RAM usage | ~120-150 MB |
| First response time | 2-5 seconds (depends on LLM provider + WiFi) |
| Subsequent responses | 1-3 seconds |
| SQLite DB size (30 days) | ~5-20 MB |
| Boot to ready | ~15-20 seconds |

**Tips for best performance:**
- Use Groq as the LLM provider (fastest free-tier inference)
- Set `gpu_mem=16` in `/boot/config.txt` to free RAM
- Keep `MAX_HISTORY_PAIRS=10` or lower
- The server is single-threaded by design — one request at a time
- WiFi latency adds ~50-200ms to each API call

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Make sure venv is activated: `source venv/bin/activate` |
| `ConfigError: LLM_API_KEY not set` | Edit `.env` and set your API key |
| Server unreachable from other devices | Check firewall: `sudo ufw allow 5000` |
| `arp-scan: permission denied` | Set up sudoers (see Network Scanning section) |
| Slow pip install | Normal on Pi Zero 2W — C extensions take time to compile |
| `MemoryError` during pip install | Add swap: `sudo dphys-swapfile swapoff && sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=512/' /etc/dphys-swapfile && sudo dphys-swapfile setup && sudo dphys-swapfile swapon` |
| Database locked errors | Ensure only one instance is running: `sudo systemctl stop jarvis` before running manually |
| WiFi drops | Add `wireless-power off` to `/etc/network/interfaces` or create `/etc/NetworkManager/conf.d/wifi-powersave-off.conf` |

## Project Structure

```
JarvisAssistant/
├── run.py                      # Main entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── README.md                   # This file
├── app/
│   ├── __init__.py             # Flask app factory
│   ├── config.py               # Configuration management
│   ├── routes.py               # All HTTP endpoints
│   ├── auth.py                 # Authentication (login/session)
│   ├── conversation_manager.py # Chat orchestration + tool routing
│   ├── llm_client.py           # Groq/HuggingFace/Gemini clients
│   ├── command_executor.py     # Bash command execution + blocklist
│   ├── web_searcher.py         # DuckDuckGo web search
│   ├── network_scanner.py      # ARP/ip neigh/proc network scan
│   ├── calendar_client.py      # Google Calendar + CalDAV
│   ├── email_client.py         # SMTP email sending
│   ├── database_manager.py     # SQLite persistence
│   ├── notes_manager.py        # Mental notes + reminders
│   ├── metrics_collector.py    # Usage metrics tracking
│   ├── system_monitor.py       # CPU/RAM/disk from /proc
│   ├── templates/              # Jinja2 HTML templates
│   │   ├── base.html
│   │   ├── login.html
│   │   ├── chat.html
│   │   ├── dashboard.html
│   │   └── settings.html
│   └── static/
│       ├── css/style.css
│       └── js/
│           ├── htmx.min.js     # Replace stub with real library
│           └── alpine.min.js   # Replace stub with real library
└── tests/                      # Property-based + unit tests
```

## License

MIT
