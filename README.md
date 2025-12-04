# Backend Service

The Backend Service is the core orchestration layer that coordinates browser automation tasks. It uses the browser_use library to execute AI-driven browser automation, connecting to the Browser Service for browser instances and the Database Service for data persistence.

## Overview

This service:
- Receives browser automation task requests
- Manages task execution using browser_use
- Connects to remote browsers via Browser Service
- Saves task outputs to Database Service
- Streams real-time updates via WebSocket

## Features

- AI-powered browser automation using browser_use
- Remote browser support via CDP URLs
- Real-time task streaming via WebSocket
- Integration with Browser and Database services
- Task lifecycle management
- Error handling and recovery

## Installation

### Prerequisites

- Python 3.11 or higher
- Google API Key (for Gemini LLM)
- Browser Service running (port 8001)
- Database Service running (port 8002)

### Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:

Create a `.env` file:
```bash
GOOGLE_API_KEY=your_google_api_key_here
BROWSER_SERVICE_URL=http://localhost:8001
DATABASE_SERVICE_URL=http://localhost:8002
BACKEND_SERVICE_PORT=8000
```

3. Install browser_use library:
```bash
pip install browser-use
```

## Usage

### Start the Service

```bash
python api_server.py
```

Or with custom port:
```bash
BACKEND_SERVICE_PORT=8000 python api_server.py
```

The service will start on `http://localhost:8000` by default.

## API Endpoints

### Health Check
```bash
GET /health
```

### Start Task
```bash
POST /tasks/start
Content-Type: application/json

{
  "task_prompt": "Search for browser automation on DuckDuckGo",
  "max_steps": 100,
  "user_id": "user123",      // Optional
  "browser_name": "firefox", // Optional: firefox, chrome, webkit
  "browser_port": 9999       // Optional
}
```

Response:
```json
{
  "success": true,
  "task_id": "507f1f77bcf86cd799439011",
  "message": "Task started successfully"
}
```

### Get Task Status
```bash
GET /tasks/{task_id}/status
```

Response:
```json
{
  "success": true,
  "status": "running",  // pending, running, completed, failed, cancelled
  "message": "Task found"
}
```

### Cancel Task
```bash
POST /tasks/{task_id}/cancel
```

### WebSocket Stream

Connect to receive real-time task updates:
```javascript
const ws = new WebSocket('ws://localhost:8000/tasks/{task_id}/stream');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

## Remote Browser Configuration

The service uses remote browsers via CDP URLs. When starting a task:

1. Requests a browser instance from Browser Service
2. Receives a CDP URL (e.g., `http://localhost:9999`)
3. Configures browser_use to use the remote browser:
```python
browser = Browser(
    headless=True,
    cdp_url="http://localhost:9999"
)
```

### Using Browser-Use Cloud Browser

You can also use browser-use cloud service:
```python
browser = Browser(
    use_cloud=True,  # Automatically provisions a cloud browser
)

# Or with advanced settings:
browser = Browser(
    cloud_profile_id='your-profile-id',
    cloud_proxy_country_code='us',
    cloud_timeout=30,
)
```

For cloud browser, you need:
- API key from [cloud.browser-use.com](https://cloud.browser-use.com/new-api-key)
- Set `BROWSER_USE_API_KEY` environment variable

### Using Custom CDP URL

You can also use any CDP URL from any provider:
```python
browser = Browser(
    cdp_url="http://remote-server:9222"
)
```

## Task Execution Flow

1. **Task Creation**: Service receives task request and creates task in Database Service
2. **Browser Acquisition**: Requests browser instance from Browser Service
3. **Agent Initialization**: Creates browser_use Agent with remote browser
4. **Execution**: Runs agent with step-by-step callbacks
5. **Streaming**: Sends updates via WebSocket and saves to Database
6. **Completion**: Saves final results and updates task status

## Stream Data Format

The service streams different types of updates:

### Task Start
```json
{
  "type": "task_start",
  "task": "Search for browser automation",
  "max_steps": 100
}
```

### Step Update
```json
{
  "type": "step",
  "step_number": 1,
  "url": "https://duckduckgo.com",
  "thinking": "I need to search...",
  "next_goal": "Click the search box",
  "actions": [...],
  "results": [...]
}
```

### Task Complete
```json
{
  "type": "task_complete",
  "is_done": true,
  "is_successful": true,
  "final_result": "Successfully searched...",
  "total_steps": 5,
  "urls_visited": ["https://duckduckgo.com"],
  "errors": []
}
```

### Error
```json
{
  "type": "error",
  "error": "Error message",
  "error_type": "ExceptionType"
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | Required | Google API key for Gemini LLM |
| `BROWSER_SERVICE_URL` | `http://localhost:8001` | Browser Service endpoint |
| `DATABASE_SERVICE_URL` | `http://localhost:8002` | Database Service endpoint |
| `BACKEND_SERVICE_PORT` | `8000` | Backend service port |
| `BROWSER_USE_API_KEY` | Optional | For cloud browser service |

## Architecture

The service uses:
- **FastAPI** for REST API and WebSocket support
- **browser_use** for AI browser automation
- **ChatGoogle** (Gemini) as the default LLM
- **httpx** for HTTP client requests to other services

## Integration Points

### Browser Service
- `POST /browser/start` - Request browser instance
- `GET /browser/{port}/connection` - Get browser CDP URL
- `POST /browser/stop` - Release browser instance

### Database Service
- `POST /tasks` - Create task
- `POST /tasks/{id}/outputs` - Save step output
- `PUT /tasks/{id}/status` - Update task status
- `GET /tasks/{id}` - Get task details
- `GET /tasks/{id}/history` - Get task history

## LLM Configuration

Default LLM is Google Gemini (`gemini-flash-latest`). To change:

1. Import a different LLM from browser_use:
```python
from browser_use import ChatOpenAI, ChatAnthropic, ChatGroq

llm = ChatOpenAI(model='gpt-4')  # or ChatAnthropic, ChatGroq, etc.
```

2. Update the agent initialization in `api_server.py`

## Troubleshooting

### Browser Connection Issues
- Verify Browser Service is running
- Check CDP URL is accessible
- Ensure browser is actually started on the port

### Task Execution Fails
- Check Google API key is set
- Verify LLM quota/limits
- Check browser_use library version
- Review logs for specific errors

### WebSocket Connection Issues
- Verify CORS settings
- Check firewall/network rules
- Ensure task ID is valid

## Development

### Project Structure
```
back-end/
├── api_server.py      # Main FastAPI server
├── app.py            # Original standalone version
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

### Testing

Test the service:
```bash
# Health check
curl http://localhost:8000/health

# Start a task
curl -X POST http://localhost:8000/tasks/start \
  -H "Content-Type: application/json" \
  -d '{
    "task_prompt": "Search for Python tutorials",
    "max_steps": 10
  }'
```

## License

Part of the Data Collector ADIA project.
