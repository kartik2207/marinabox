# Marinabox

Containerized sandboxes for AI agents

## Overview

MarinaBox is a toolkit for creating and managing secure, isolated environments for AI agents. It provides:

### Core Features

1. **Secure Sandboxed Environments**
   - Run isolated browser and desktop sessions locally or cloud
   - Perfect for AI agent tasks(Computer Use) and browser automation

2. **Comprehensive SDK & CLI**
   - Python SDK for programmatic control
   - Command-line interface for session management
   - Real-time monitoring and control capabilities
   - Integration with popular automation tools (Playwright, Selenium)

3. **Interactive UI Dashboard**
   - Live session viewing and control
   - Session recording and playback
   - Session management interface

### Additional Features

- **Cloud Integration**: Deploy sandboxes to major cloud providers(coming soon)
- **Multi-session Management**: Run multiple isolated environments simultaneously
- **Session Tagging**: Organize and track sessions with custom tags

## Prerequisites

- Docker
- Python 3.12 or higher
- pip (Python package installer)

## Important Note

The provided Docker images are built for Mac ARM64 architecture (Apple Silicon). For other architectures:

1. Clone the sandbox repository:
```bash
git clone https://github.com/marinabox/marinabox-sandbox
```

2. Build the images with your target platform:
```bash
docker build --platform <your-platform> -t marinabox/marinabox-browser .
docker build --platform <your-platform> -t marinabox/marinabox-desktop .
```

## Installation

1. First, ensure you have Docker installed on your system. If not, [install Docker](https://docs.docker.com/get-docker/) for your operating system.

2. Pull the required Docker images:
```bash
docker pull marinabox/marinabox-browser:latest
docker pull marinabox/marinabox-desktop:latest
```

3. Install the Marinabox package:
```bash
pip install marinabox
```

## Usage Example

Here's a basic example of how to use the Marinabox SDK:

```python
from marinabox import MarinaboxSDK

# Initialize the SDK
mb = MarinaboxSDK()

# Set Anthropic API key
mb.set_anthropic_key(ANTHROPIC_API_KEY)

# Create a new session
session = mb.create_session(env_type="browser", tag="my-session")
print(f"Created session: {session.session_id}")

# List active sessions
sessions = mb.list_sessions()
for s in sessions:
    print(f"Active session: {s.session_id} (Tag: {s.tag})")

# Execute a computer use command
mb.computer_use_command("my-session", "Navigate to https://x.ai")
