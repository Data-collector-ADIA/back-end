"""
Backend Service - Orchestrates browser automation tasks
Connects to Browser Service and Database Service via HTTP REST APIs.
Uses browser_use library with remote browser support.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import AsyncGenerator, Dict, Optional

import httpx
from browser_use import Agent, Browser, ChatGoogle
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Backend Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs (can be overridden via environment variables)
BROWSER_SERVICE_URL = os.getenv("BROWSER_SERVICE_URL", "http://localhost:8001")
DATABASE_SERVICE_URL = os.getenv("DATABASE_SERVICE_URL", "http://localhost:8002")

# HTTP client for service communication
http_client = httpx.AsyncClient(timeout=30.0)

# Active tasks tracking
active_tasks: Dict[str, asyncio.Task] = {}


# Pydantic models
class StartTaskRequest(BaseModel):
    task_prompt: str
    max_steps: int = 100
    user_id: Optional[str] = None
    browser_name: str = "firefox"
    browser_port: int = 9999


class StartTaskResponse(BaseModel):
    success: bool
    task_id: Optional[str] = None
    message: str


class TaskStatusResponse(BaseModel):
    success: bool
    status: str
    message: str


def _suppress_logging():
    """Suppress all logging output to prevent printing"""
    os.environ['BROWSER_USE_LOGGING_LEVEL'] = 'critical'
    
    # Create a null handler that discards all logs
    null_handler = logging.NullHandler()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = [null_handler]
    root_logger.setLevel(logging.CRITICAL)
    
    # Configure browser_use loggers
    for logger_name in ['browser_use', 'browser_use.agent', 'browser_use.agent.service', 
                       'browser_use.tools', 'browser_use.browser', 'cdp_use']:
        logger = logging.getLogger(logger_name)
        logger.handlers = [null_handler]
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False


def _format_step_data(agent: Agent, step_number: int) -> dict:
    """Format step data as JSON-serializable dictionary"""
    step_data = {
        'type': 'step',
        'step_number': step_number,
        'timestamp': None,
        'url': None,
        'actions': [],
        'results': [],
        'evaluation': None,
        'memory': None,
        'next_goal': None,
        'thinking': None,
        'error': None,
    }
    
    # Get the last history item if available
    if agent.history and agent.history.history:
        last_item = agent.history.history[-1]
        
        # Extract URL
        if last_item.state and hasattr(last_item.state, 'url'):
            step_data['url'] = last_item.state.url
        
        # Extract metadata timestamp
        if last_item.metadata:
            step_data['timestamp'] = last_item.metadata.step_end_time
        
        # Extract model output (thinking, evaluation, memory, goals)
        if last_item.model_output:
            if hasattr(last_item.model_output, 'current_state'):
                current_state = last_item.model_output.current_state
                if current_state:
                    step_data['thinking'] = current_state.thinking if hasattr(current_state, 'thinking') else None
                    step_data['evaluation'] = current_state.evaluation_previous_goal if hasattr(current_state, 'evaluation_previous_goal') else None
                    step_data['memory'] = current_state.memory if hasattr(current_state, 'memory') else None
                    step_data['next_goal'] = current_state.next_goal if hasattr(current_state, 'next_goal') else None
            
            # Extract actions
            if last_item.model_output.action:
                for action in last_item.model_output.action:
                    action_dict = action.model_dump(exclude_unset=True) if hasattr(action, 'model_dump') else {}
                    step_data['actions'].append(action_dict)
        
        # Extract results
        if last_item.result:
            for result in last_item.result:
                result_dict = {
                    'extracted_content': result.extracted_content,
                    'error': result.error,
                    'is_done': result.is_done,
                    'success': result.success,
                    'long_term_memory': result.long_term_memory,
                }
                if result.attachments:
                    result_dict['attachments'] = result.attachments
                step_data['results'].append(result_dict)
    
    return step_data


async def run_agent_task(
    task_id: str,
    prompt: str,
    max_steps: int,
    browser_cdp_url: str,
    websocket: Optional[WebSocket] = None
):
    """
    Run an agent task and stream updates
    """
    browser = None
    try:
        # Suppress logging
        _suppress_logging()
        
        # Initialize browser with remote CDP URL
        browser = Browser(
            headless=True,
            cdp_url=browser_cdp_url
        )
        
        # Initialize agent with Gemini
        llm = ChatGoogle(model='gemini-flash-latest')
        agent = Agent(
            task=prompt,
            llm=llm,
            browser=browser,
        )
        
        # Update task status to running
        await http_client.put(
            f"{DATABASE_SERVICE_URL}/tasks/{task_id}/status",
            json={"status": "running", "final_result": None}
        )
        
        # Send initial task start
        if websocket:
            await websocket.send_json({
                'type': 'task_start',
                'task': prompt,
                'max_steps': max_steps,
            })
        
        await save_task_output(task_id, {
            'type': 'task_start',
            'task': prompt,
            'max_steps': max_steps,
        }, 'task_start', 0)
        
        # Define callback to send step data
        async def on_step_end(agent_instance: Agent):
            step_number = agent_instance.state.n_steps
            step_data = _format_step_data(agent_instance, step_number)
            
            # Send via WebSocket if available
            if websocket:
                try:
                    await websocket.send_json(step_data)
                except Exception as e:
                    logger.error(f"Error sending WebSocket message: {e}")
            
            # Save to database
            await save_task_output(task_id, step_data, 'step', step_number)
        
        # Run the agent
        history = await agent.run(
            max_steps=max_steps,
            on_step_end=on_step_end,
        )
        
        # Get final result
        final_result = history.final_result()
        is_done = history.is_done()
        is_successful = history.is_successful()
        
        completion_data = {
            'type': 'task_complete',
            'is_done': is_done,
            'is_successful': is_successful,
            'final_result': final_result,
            'total_steps': history.number_of_steps(),
            'urls_visited': history.urls(),
            'errors': history.errors(),
        }
        
        # Send completion via WebSocket
        if websocket:
            try:
                await websocket.send_json(completion_data)
            except Exception as e:
                logger.error(f"Error sending completion via WebSocket: {e}")
        
        # Save completion to database
        await save_task_output(task_id, completion_data, 'task_complete', history.number_of_steps())
        
        # Update task status
        status = "completed" if is_successful else "failed"
        await http_client.put(
            f"{DATABASE_SERVICE_URL}/tasks/{task_id}/status",
            json={
                "status": status,
                "final_result": json.dumps(completion_data)
            }
        )
        
    except Exception as e:
        logger.error(f"Error running agent task {task_id}: {e}", exc_info=True)
        
        error_data = {
            'type': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
        }
        
        # Send error via WebSocket
        if websocket:
            try:
                await websocket.send_json(error_data)
            except Exception:
                pass
        
        # Save error to database
        await save_task_output(task_id, error_data, 'error', 0)
        
        # Update task status
        await http_client.put(
            f"{DATABASE_SERVICE_URL}/tasks/{task_id}/status",
            json={
                "status": "failed",
                "final_result": json.dumps(error_data)
            }
        )
    finally:
        # Cleanup browser
        if browser:
            try:
                await browser.stop()
            except Exception:
                pass
        
        # Remove from active tasks
        if task_id in active_tasks:
            del active_tasks[task_id]


async def save_task_output(task_id: str, step_data: dict, output_type: str, step_number: int):
    """Save task output to database service"""
    try:
        await http_client.post(
            f"{DATABASE_SERVICE_URL}/tasks/{task_id}/outputs",
            json={
                "task_id": task_id,
                "step_data": json.dumps(step_data, default=str),
                "output_type": output_type,
                "step_number": step_number
            }
        )
    except Exception as e:
        logger.error(f"Error saving task output: {e}")


async def get_browser_connection(browser_name: str, browser_port: int) -> str:
    """Get or start a browser and return CDP URL"""
    try:
        # Check if browser is already running
        response = await http_client.get(f"{BROWSER_SERVICE_URL}/browser/{browser_port}/connection")
        conn_data = response.json()
        
        if conn_data.get("running"):
            return conn_data["cdp_url"]
        
        # Start browser
        start_response = await http_client.post(
            f"{BROWSER_SERVICE_URL}/browser/start",
            json={
                "browser_name": browser_name,
                "port": browser_port
            }
        )
        
        start_data = start_response.json()
        if not start_data.get("success"):
            raise Exception(f"Failed to start browser: {start_data.get('message')}")
        
        return start_data["cdp_url"]
        
    except Exception as e:
        logger.error(f"Error getting browser connection: {e}")
        raise


# API Endpoints
@app.post("/tasks/start", response_model=StartTaskResponse)
async def start_task(request: StartTaskRequest):
    """Start a new browser automation task"""
    try:
        # Create task in database
        db_response = await http_client.post(
            f"{DATABASE_SERVICE_URL}/tasks",
            json={
                "task_prompt": request.task_prompt,
                "max_steps": request.max_steps,
                "user_id": request.user_id
            }
        )
        
        if db_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to create task in database")
        
        db_data = db_response.json()
        task_id = db_data.get("task_id")
        
        if not task_id:
            raise HTTPException(status_code=500, detail="Task ID not returned from database")
        
        # Get browser connection
        browser_cdp_url = await get_browser_connection(request.browser_name, request.browser_port)
        
        # Start task asynchronously
        task = asyncio.create_task(
            run_agent_task(task_id, request.task_prompt, request.max_steps, browser_cdp_url)
        )
        active_tasks[task_id] = task
        
        return StartTaskResponse(
            success=True,
            task_id=task_id,
            message="Task started successfully"
        )
    except Exception as e:
        logger.error(f"Error starting task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/tasks/{task_id}/stream")
async def stream_task(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for streaming task updates"""
    await websocket.accept()
    
    try:
        # Get task from database
        task_response = await http_client.get(f"{DATABASE_SERVICE_URL}/tasks/{task_id}")
        if task_response.status_code != 200:
            await websocket.close(code=1008, reason="Task not found")
            return
        
        task_data = task_response.json()
        
        # Get browser connection
        browser_cdp_url = await get_browser_connection("firefox", 9999)
        
        # Run task and stream updates
        await run_agent_task(
            task_id,
            task_data["task_prompt"],
            task_data["max_steps"],
            browser_cdp_url,
            websocket
        )
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket stream: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass


@app.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get task status"""
    try:
        response = await http_client.get(f"{DATABASE_SERVICE_URL}/tasks/{task_id}")
        
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_data = response.json()
        
        return TaskStatusResponse(
            success=True,
            status=task_data.get("status", "unknown"),
            message="Task found"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running task"""
    try:
        if task_id in active_tasks:
            task = active_tasks[task_id]
            task.cancel()
            del active_tasks[task_id]
            
            # Update task status
            await http_client.put(
                f"{DATABASE_SERVICE_URL}/tasks/{task_id}/status",
                json={"status": "cancelled"}
            )
            
            return {"success": True, "message": "Task cancelled"}
        else:
            raise HTTPException(status_code=404, detail="Task not found or not running")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "backend-service"}


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await http_client.aclose()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BACKEND_SERVICE_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

