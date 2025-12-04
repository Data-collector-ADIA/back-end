# Backend Service

The Backend Service is the core orchestration layer that coordinates browser automation tasks. It uses the browser_use library to execute AI-driven browser automation, connecting to the Browser Service and Database Service via gRPC.

## Overview

This service:
- Receives browser automation task requests via gRPC
- Manages task execution using browser_use
- Connects to remote browsers via Browser Service (gRPC)
- Saves task outputs to Database Service (gRPC)
- Coordinates all service interactions

## Features

- AI-powered browser automation using browser_use
- Remote browser support via CDP URLs
- gRPC-based communication for better performance
- Integration with Browser and Database services
- Task lifecycle management
- Error handling and recovery

## Installation

### Prerequisites

- Python 3.11 or higher
- Google API Key (for Gemini LLM)
- Browser Service running (port 50051)
- Database Service running (port 50052)

### Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:

Create a `.env` file:
```bash
GOOGLE_API_KEY=your_google_api_key_here
BROWSER_SERVICE_HOST=localhost
BROWSER_SERVICE_PORT=50051
DATABASE_SERVICE_HOST=localhost
DATABASE_SERVICE_PORT=50052
BACKEND_SERVICE_PORT=50050
```

Get your Google API key from: https://aistudio.google.com/apikey

3. Generate protobuf files (first time only):
```bash
cd ../shared
python generate_protos.py
cd ../back-end
```

## Configuration

### Environment Variables

```bash
BACKEND_SERVICE_HOST=localhost          # Host to bind to (default: localhost)
BACKEND_SERVICE_PORT=50050              # gRPC port (default: 50050)
BROWSER_SERVICE_HOST=localhost          # Browser Service host (default: localhost)
BROWSER_SERVICE_PORT=50051              # Browser Service port (default: 50051)
DATABASE_SERVICE_HOST=localhost         # Database Service host (default: localhost)
DATABASE_SERVICE_PORT=50052             # Database Service port (default: 50052)
GOOGLE_API_KEY=your_api_key            # Required: Google Gemini API key
```

### Default Ports

- **gRPC Service**: `50050`
- **Browser Service**: `50051` (remote)
- **Database Service**: `50052` (remote)

## Usage

### Start the Service

**For Testing (Single Machine):**
```bash
# Run in a screen session
screen -S backend-service
python server.py
# Press Ctrl+A then D to detach
```

**For Production (Separate Machine):**
```bash
export BACKEND_SERVICE_HOST=0.0.0.0
export BACKEND_SERVICE_PORT=50050
export BROWSER_SERVICE_HOST=browser-service-host
export BROWSER_SERVICE_PORT=50051
export DATABASE_SERVICE_HOST=database-service-host
export DATABASE_SERVICE_PORT=50052
export GOOGLE_API_KEY=your_api_key
python server.py
```

### gRPC Methods

The service exposes these gRPC methods:

#### StartTask
Start a new browser automation task.

**Request:**
```protobuf
message StartTaskRequest {
  string task_prompt = 1;
  int32 max_steps = 2;
  string user_id = 3;
  string browser_name = 4;  // Optional: firefox, webkit, chrome
  int32 browser_port = 5;   // Optional: specific browser port
}
```

**Response:**
```protobuf
message StartTaskResponse {
  bool success = 1;
  string task_id = 2;
  string message = 3;
}
```

#### GetTaskStatus
Get current task status.

**Request:**
```protobuf
message GetTaskStatusRequest {
  string task_id = 1;
}
```

**Response:**
```protobuf
message GetTaskStatusResponse {
  bool success = 1;
  string status = 2;  // running, completed, failed, cancelled
  string message = 3;
}
```

#### CancelTask
Cancel a running task.

**Request:**
```protobuf
message CancelTaskRequest {
  string task_id = 1;
}
```

**Response:**
```protobuf
message CancelTaskResponse {
  bool success = 1;
  string message = 2;
}
```

## Architecture

The service uses:
- **gRPC** for RPC communication
- **browser_use** for browser automation
- **Google Gemini** (via browser_use) for AI agent
- **Protocol Buffers** for type-safe messages

## Integration with Other Services

### Browser Service
The Backend Service connects to Browser Service via gRPC to:
1. Request a browser instance (`StartBrowser`)
2. Get CDP URL for browser connection (`GetBrowserConnection`)
3. Release browser when done (`StopBrowser`)

### Database Service
The Backend Service connects to Database Service via gRPC to:
1. Create tasks before execution (`CreateTask`)
2. Save step-by-step outputs during execution (`SaveTaskOutput`)
3. Update task status as execution progresses (`UpdateTaskStatus`)
4. Store final results when tasks complete

## Task Execution Flow

1. **Receive Task Request** - Frontend calls `StartTask` via gRPC
2. **Create Task Record** - Backend calls Database Service `CreateTask`
3. **Request Browser** - Backend calls Browser Service `StartBrowser`
4. **Get CDP URL** - Backend receives CDP URL from Browser Service
5. **Initialize Agent** - Backend creates browser_use Agent with CDP URL
6. **Execute Task** - Agent runs browser automation
7. **Save Outputs** - On each step, Backend calls Database Service `SaveTaskOutput`
8. **Update Status** - On completion, Backend calls Database Service `UpdateTaskStatus`
9. **Cleanup** - Backend calls Browser Service `StopBrowser`

## Testing

### Using grpcurl

```bash
# List available services
grpcurl -plaintext localhost:50050 list

# Start task
grpcurl -plaintext -d '{
  "task_prompt": "Search for browser automation on DuckDuckGo",
  "max_steps": 10,
  "user_id": "test_user",
  "browser_name": "firefox"
}' localhost:50050 backend_service.BackendService/StartTask

# Get task status
grpcurl -plaintext -d '{"task_id": "your_task_id"}' \
  localhost:50050 backend_service.BackendService/GetTaskStatus
```

## Troubleshooting

### Cannot Connect to Browser Service

```bash
# Verify Browser Service is running
grpcurl -plaintext localhost:50051 list

# Check environment variables
echo $BROWSER_SERVICE_HOST
echo $BROWSER_SERVICE_PORT
```

### Cannot Connect to Database Service

```bash
# Verify Database Service is running
grpcurl -plaintext localhost:50052 list

# Check environment variables
echo $DATABASE_SERVICE_HOST
echo $DATABASE_SERVICE_PORT
```

### Google API Key Issues

```bash
# Verify API key is set
cat .env | grep GOOGLE_API_KEY

# Test API key (if you have curl)
curl "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key=$GOOGLE_API_KEY"
```

### Port Already in Use

```bash
# Check what's using the port
lsof -i :50050  # Linux/Mac
netstat -an | findstr 50050  # Windows

# Change port
export BACKEND_SERVICE_PORT=50051
python server.py
```

### Protobuf Import Errors

```bash
# Regenerate protobuf files
cd ../shared
python generate_protos.py
```

### Task Execution Fails

- Check all services are running
- Verify browser service can start browsers
- Check Google API quota/limits
- Review service logs for specific errors

## Development

### Project Structure
```
back-end/
├── server.py              # gRPC server
├── app.py                 # Core browser automation logic
├── browser_use/           # browser_use library
├── requirements.txt       # Python dependencies
├── QUICKSTART.md          # Quick start guide
└── README.md              # This file
```

### Running in Development

```bash
# Start service
python server.py

# In another terminal, test with grpcurl
grpcurl -plaintext localhost:50050 list
```

## Deployment

### Single Machine (Testing)

Run in a screen session:
```bash
screen -S backend-service
python server.py
```

### Separate Machine (Production)

1. Install dependencies
2. Set up environment variables
3. Configure service addresses
4. Run service: `python server.py`
5. Expose port 50050

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for a quick setup guide.

## License

Part of the Data Collector ADIA project.
