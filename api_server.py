import asyncio
import json
import logging
import os
from typing import AsyncGenerator

import grpc
from dotenv import load_dotenv
from browser_use import Agent, Browser, ChatGoogle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import browser service protobuf
try:
    import browser_service_pb2
    import browser_service_pb2_grpc
    PROTOBUF_AVAILABLE = True
except ImportError:
    logging.warning("Protobuf files not found. Browser service integration may not work.")
    PROTOBUF_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Service addresses
BROWSER_SERVICE_HOST = os.getenv("BROWSER_SERVICE_HOST", "localhost")
BROWSER_SERVICE_PORT = int(os.getenv("BROWSER_SERVICE_PORT", "50061"))

# Create FastAPI app
app = FastAPI(
    title="Browser Automation API",
    description="API for browser automation with streaming results",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global browser channel
_browser_channel = None

def get_browser_stub():
    global _browser_channel
    if _browser_channel is None:
        _browser_channel = grpc.insecure_channel(f"{BROWSER_SERVICE_HOST}:{BROWSER_SERVICE_PORT}")
    return browser_service_pb2_grpc.BrowserServiceStub(_browser_channel)


def _suppress_logging():
    """Suppress all logging output to prevent printing"""
    os.environ['BROWSER_USE_LOGGING_LEVEL'] = 'critical'
    
    # Create a null handler that discards all logs
    null_handler = logging.NullHandler()
    
    # Configure browser_use loggers
    for logger_name in ['browser_use', 'browser_use.agent', 'browser_use.agent.service', 
                       'browser_use.tools', 'browser_use.browser', 'cdp_use']:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = [null_handler]
        logger_instance.setLevel(logging.CRITICAL)
        logger_instance.propagate = False


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
        
        # Extract model output
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


async def run_agent_stream(prompt: str, max_steps: int = 100) -> AsyncGenerator[dict, None]:
    """
    Run an agent with the given prompt and stream updates as JSON.
    
    Args:
        prompt: The task prompt for the agent
        max_steps: Maximum number of steps the agent can take
        
    Yields:
        dict: JSON-serializable dictionaries containing step updates
    """
    # Suppress all logging output
    _suppress_logging()
    
    browser = None
    cdp_url = None
    browser_port = 9999
    
    try:
        # Get browser connection via gRPC
        browser_client = get_browser_stub()
        
        # Start browser
        start_request = browser_service_pb2.StartBrowserRequest(
            browser_name="chrome",
            port=browser_port
        )
        browser_response = browser_client.StartBrowser(start_request)
        
        if not browser_response.success:
            raise Exception(f"Failed to start browser: {browser_response.message}")
        
        cdp_url = browser_response.cdp_url
        logger.info(f"Browser started, CDP URL: {cdp_url}")
        
        # Initialize agent
        llm = ChatGoogle(model='gemini-flash-latest')
        browser = Browser(headless=True, cdp_url=cdp_url)
        
        # Start the browser
        await browser.start()
        
        agent = Agent(
            task=prompt,
            llm=llm,
            browser=browser,
            include_attributes=['href', 'text', 'type', 'name', 'id'],
            max_failures=2,
            retry_delay=10,
        )
        
        # Queue to collect step updates
        update_queue = asyncio.Queue()
        agent_task = None
        
        # Define callback to queue step data
        async def on_step_end(agent_instance: Agent):
            step_data = _format_step_data(agent_instance, agent_instance.state.n_steps)
            await update_queue.put(step_data)
        
        # Task to run the agent
        async def run_agent():
            try:
                history = await agent.run(
                    max_steps=max_steps,
                    on_step_end=on_step_end,
                )
                
                # Signal completion
                final_result = history.final_result()
                is_done = history.is_done()
                is_successful = history.is_successful()
                
                await update_queue.put({
                    'type': 'task_complete',
                    'is_done': is_done,
                    'is_successful': is_successful,
                    'final_result': final_result,
                    'total_steps': history.number_of_steps(),
                    'urls_visited': history.urls(),
                    'errors': history.errors(),
                })
            except Exception as e:
                await update_queue.put({
                    'type': 'error',
                    'error': str(e),
                    'error_type': type(e).__name__,
                })
            finally:
                # Signal end of updates
                await update_queue.put(None)
        
        # Yield initial task info
        yield {
            'type': 'task_start',
            'task': prompt,
            'max_steps': max_steps,
        }
        
        # Start agent task
        agent_task = asyncio.create_task(run_agent())
        
        # Stream updates as they arrive
        while True:
            update = await update_queue.get()
            if update is None:
                break
            yield update
        
        # Wait for agent task to complete
        if agent_task:
            await agent_task
            
    except Exception as e:
        # Yield error information
        yield {
            'type': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
        }
        logger.error(f"Error in agent stream: {e}", exc_info=True)
    
    finally:
        # Cleanup browser
        if browser:
            try:
                await browser.stop()
            except Exception:
                pass
        
        # Stop browser via service
        if cdp_url:
            try:
                browser_client = get_browser_stub()
                stop_request = browser_service_pb2.StopBrowserRequest(port=browser_port)
                browser_client.StopBrowser(stop_request)
            except Exception as e:
                logger.error(f"Error stopping browser via service: {e}")


# Request models
class RunTaskRequest(BaseModel):
    prompt: str
    max_steps: int = 100

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Search for 'browser automation' on DuckDuckGo",
                "max_steps": 10
            }
        }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Browser Automation API",
        "version": "1.0.0",
        "endpoints": {
            "run_task": "/api/v1/run",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/v1/run")
async def run_task(request: RunTaskRequest):
    """
    Run a browser automation task and stream results as Server-Sent Events.
    
    The response is a stream of JSON objects, each representing a step or event:
    - task_start: Initial task information
    - step: Step-by-step progress with actions, results, and agent thinking
    - task_complete: Final results when task finishes
    - error: Error information if something fails
    
    Example:
        ```python
        import requests
        
        response = requests.post(
            "http://localhost:8000/api/v1/run",
            json={"prompt": "Search for AI news", "max_steps": 5},
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8').replace('data: ', ''))
                print(data)
        ```
    """
    try:
        async def event_generator():
            async for update in run_agent_stream(request.prompt, request.max_steps):
                # Format as Server-Sent Event
                yield f"data: {json.dumps(update, default=str)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        logger.error(f"Error in run_task endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_SERVER_PORT", "8000"))
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
