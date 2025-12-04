# Backend Service - Quick Start

Get the Backend Service running quickly for testing or production deployment.

## Prerequisites

- Python 3.11+
- Google API Key (for Gemini LLM)
- Browser Service running (port 50051)
- Database Service running (port 50052)

## Quick Setup

### 1. Install Dependencies

```bash
cd back-end
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

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

### 3. Generate Protobuf Files (First Time Only)

```bash
cd ../shared
python generate_protos.py
cd ../back-end
```

## Running the Service

### For Testing (Single Machine)

Run in a screen session:

```bash
# Create a new screen session
screen -S backend-service

# Start the service
python server.py

# Detach: Press Ctrl+A then D
# Reattach: screen -r backend-service
```

### For Production (Separate Machine)

```bash
# Set environment variables
export BACKEND_SERVICE_HOST=0.0.0.0
export BACKEND_SERVICE_PORT=50050
export BROWSER_SERVICE_HOST=browser-service-host
export BROWSER_SERVICE_PORT=50051
export DATABASE_SERVICE_HOST=database-service-host
export DATABASE_SERVICE_PORT=50052
export GOOGLE_API_KEY=your_api_key

# Run the service
python server.py
```

Or use a process manager like systemd, supervisor, or PM2.

## Verify Service is Running

### Check gRPC Service

```bash
# Using grpcurl (if installed)
grpcurl -plaintext localhost:50050 list
```

Expected output:
```
backend_service.BackendService
```

### Test Service Connectivity

Verify connections to other services:

```bash
# Test Browser Service connection
grpcurl -plaintext localhost:50051 list

# Test Database Service connection
grpcurl -plaintext localhost:50052 list
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

## Service Endpoints (gRPC)

The service exposes these gRPC methods:

- `StartTask` - Start a new browser automation task
- `GetTaskStatus` - Get current task status
- `CancelTask` - Cancel a running task

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

## Next Steps

- See [README.md](README.md) for detailed documentation
- Configure for production deployment
- Set up monitoring and logging
- Test task execution

