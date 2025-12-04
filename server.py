"""
Backend Service - gRPC server for orchestrating browser automation tasks
Connects to Browser Service and Database Service via gRPC.
Uses browser_use library with remote browser support.
"""

import asyncio
import json
import logging
import os
from concurrent import futures
from typing import Optional

import grpc
from browser_use import Agent, Browser, ChatGoogle
from dotenv import load_dotenv

# Import generated protobuf code
try:
    import backend_service_pb2
    import backend_service_pb2_grpc
except ImportError:
    logging.warning("Protobuf files not found. Using placeholder classes.")

# Load environment variables
load_dotenv()

# Service addresses (gRPC)
BROWSER_SERVICE_HOST = os.getenv("BROWSER_SERVICE_HOST", "localhost")
BROWSER_SERVICE_PORT = int(os.getenv("BROWSER_SERVICE_PORT", "50051"))
DATABASE_SERVICE_HOST = os.getenv("DATABASE_SERVICE_HOST", "localhost")
DATABASE_SERVICE_PORT = int(os.getenv("DATABASE_SERVICE_PORT", "50052"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Active tasks tracking
active_tasks = {}


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


async def run_agent_task(
    task_id: str,
    task_prompt: str,
    max_steps: int,
    browser_name: str = "firefox",
    browser_port: int = 9999
):
    """Run browser automation task"""
    _suppress_logging()
    
    try:
        # Get browser connection via gRPC
        browser_channel = grpc.insecure_channel(f"{BROWSER_SERVICE_HOST}:{BROWSER_SERVICE_PORT}")
        browser_client = None
        try:
            import browser_service_pb2_grpc
            browser_client = browser_service_pb2_grpc.BrowserServiceStub(browser_channel)
            
            # Start browser
            import browser_service_pb2
            start_request = browser_service_pb2.StartBrowserRequest(
                browser_name=browser_name,
                port=browser_port
            )
            browser_response = browser_client.StartBrowser(start_request)
            
            if not browser_response.success:
                raise Exception(f"Failed to start browser: {browser_response.message}")
            
            cdp_url = browser_response.cdp_url
            logger.info(f"Browser started, CDP URL: {cdp_url}")
        except Exception as e:
            logger.error(f"Error getting browser: {e}")
            raise
        
        # Get database client
        db_channel = grpc.insecure_channel(f"{DATABASE_SERVICE_HOST}:{DATABASE_SERVICE_PORT}")
        db_client = None
        try:
            import database_service_pb2_grpc
            db_client = database_service_pb2_grpc.DatabaseServiceStub(db_channel)
            
            # Update task status to running
            import database_service_pb2
            status_request = database_service_pb2.UpdateTaskStatusRequest(
                task_id=task_id,
                status="running"
            )
            db_client.UpdateTaskStatus(status_request)
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
        
        # Initialize agent
        llm = ChatGoogle(model='gemini-flash-latest')
        browser = Browser(headless=True, cdp_url=cdp_url)
        agent = Agent(
            task=task_prompt,
            llm=llm,
            browser=browser,
        )
        
        # Define callback to save step data
        async def on_step_end(agent_instance: Agent):
            step_data = _format_step_data(agent_instance, agent_instance.state.n_steps)
            step_json = json.dumps(step_data, default=str)
            
            # Save to database via gRPC
            if db_client:
                try:
                    output_request = database_service_pb2.SaveTaskOutputRequest(
                        task_id=task_id,
                        step_data=step_json,
                        output_type="step",
                        step_number=agent_instance.state.n_steps
                    )
                    db_client.SaveTaskOutput(output_request)
                except Exception as e:
                    logger.error(f"Error saving step output: {e}")
        
        # Run agent
        history = await agent.run(
            max_steps=max_steps,
            on_step_end=on_step_end,
        )
        
        # Get final result
        final_result = history.final_result()
        is_done = history.is_done()
        is_successful = history.is_successful()
        
        # Save final result
        final_data = {
            'type': 'task_complete',
            'is_done': is_done,
            'is_successful': is_successful,
            'final_result': final_result,
            'total_steps': history.number_of_steps(),
            'urls_visited': history.urls(),
            'errors': history.errors(),
        }
        
        if db_client:
            try:
                # Save final output
                output_request = database_service_pb2.SaveTaskOutputRequest(
                    task_id=task_id,
                    step_data=json.dumps(final_data, default=str),
                    output_type="task_complete",
                    step_number=history.number_of_steps()
                )
                db_client.SaveTaskOutput(output_request)
                
                # Update task status
                status_request = database_service_pb2.UpdateTaskStatusRequest(
                    task_id=task_id,
                    status="completed" if is_successful else "failed",
                    final_result=json.dumps(final_data, default=str)
                )
                db_client.UpdateTaskStatus(status_request)
            except Exception as e:
                logger.error(f"Error saving final result: {e}")
        
        # Cleanup browser
        try:
            if browser_client:
                stop_request = browser_service_pb2.StopBrowserRequest(port=browser_port)
                browser_client.StopBrowser(stop_request)
        except Exception as e:
            logger.error(f"Error stopping browser: {e}")
        
        # Cleanup browser
        if browser:
            try:
                await browser.stop()
            except Exception:
                pass
        
        browser_channel.close()
        db_channel.close()
        
    except Exception as e:
        logger.error(f"Error running task: {e}", exc_info=True)
        
        # Update task status to failed
        try:
            db_channel = grpc.insecure_channel(f"{DATABASE_SERVICE_HOST}:{DATABASE_SERVICE_PORT}")
            import database_service_pb2_grpc, database_service_pb2
            db_client = database_service_pb2_grpc.DatabaseServiceStub(db_channel)
            
            error_data = {
                'type': 'error',
                'error': str(e),
                'error_type': type(e).__name__,
            }
            
            output_request = database_service_pb2.SaveTaskOutputRequest(
                task_id=task_id,
                step_data=json.dumps(error_data, default=str),
                output_type="error",
                step_number=0
            )
            db_client.SaveTaskOutput(output_request)
            
            status_request = database_service_pb2.UpdateTaskStatusRequest(
                task_id=task_id,
                status="failed"
            )
            db_client.UpdateTaskStatus(status_request)
            db_channel.close()
        except Exception as e2:
            logger.error(f"Error updating failed status: {e2}")


class BackendServiceServicer:
    """gRPC servicer for Backend Service"""
    
    def StartTask(self, request, context):
        """Start a new browser automation task"""
        try:
            task_id = None
            
            # Create task in database via gRPC
            try:
                db_channel = grpc.insecure_channel(f"{DATABASE_SERVICE_HOST}:{DATABASE_SERVICE_PORT}")
                import database_service_pb2_grpc, database_service_pb2
                db_client = database_service_pb2_grpc.DatabaseServiceStub(db_channel)
                
                create_request = database_service_pb2.CreateTaskRequest(
                    task_prompt=request.task_prompt,
                    max_steps=request.max_steps,
                    user_id=request.user_id or "default"
                )
                create_response = db_client.CreateTask(create_request)
                db_channel.close()
                
                if not create_response.success:
                    raise Exception(f"Failed to create task: {create_response.message}")
                
                task_id = create_response.task_id
            except Exception as e:
                logger.error(f"Error creating task in database: {e}")
                raise
            
            # Start task execution in background
            browser_name = request.browser_name or "firefox"
            browser_port = request.browser_port or 9999
            
            task = asyncio.create_task(
                run_agent_task(
                    task_id=task_id,
                    task_prompt=request.task_prompt,
                    max_steps=request.max_steps,
                    browser_name=browser_name,
                    browser_port=browser_port
                )
            )
            active_tasks[task_id] = task
            
            response = backend_service_pb2.StartTaskResponse()
            response.success = True
            response.task_id = task_id
            response.message = "Task started successfully"
            return response
            
        except Exception as e:
            logger.error(f"Error starting task: {e}", exc_info=True)
            response = backend_service_pb2.StartTaskResponse()
            response.success = False
            response.message = str(e)
            return response
    
    def GetTaskStatus(self, request, context):
        """Get task status"""
        try:
            # Get task from database via gRPC
            db_channel = grpc.insecure_channel(f"{DATABASE_SERVICE_HOST}:{DATABASE_SERVICE_PORT}")
            import database_service_pb2_grpc, database_service_pb2
            db_client = database_service_pb2_grpc.DatabaseServiceStub(db_channel)
            
            get_request = database_service_pb2.GetTaskRequest(task_id=request.task_id)
            get_response = db_client.GetTask(get_request)
            db_channel.close()
            
            response = backend_service_pb2.GetTaskStatusResponse()
            if get_response.success:
                response.success = True
                response.status = get_response.task.status
                response.message = f"Task status: {get_response.task.status}"
            else:
                response.success = False
                response.status = "unknown"
                response.message = "Task not found"
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}", exc_info=True)
            response = backend_service_pb2.GetTaskStatusResponse()
            response.success = False
            response.status = "unknown"
            response.message = str(e)
            return response
    
    def CancelTask(self, request, context):
        """Cancel a running task"""
        try:
            task_id = request.task_id
            
            if task_id in active_tasks:
                task = active_tasks[task_id]
                task.cancel()
                del active_tasks[task_id]
            
            # Update task status in database
            try:
                db_channel = grpc.insecure_channel(f"{DATABASE_SERVICE_HOST}:{DATABASE_SERVICE_PORT}")
                import database_service_pb2_grpc, database_service_pb2
                db_client = database_service_pb2_grpc.DatabaseServiceStub(db_channel)
                
                status_request = database_service_pb2.UpdateTaskStatusRequest(
                    task_id=task_id,
                    status="cancelled"
                )
                db_client.UpdateTaskStatus(status_request)
                db_channel.close()
            except Exception as e:
                logger.error(f"Error updating cancelled status: {e}")
            
            response = backend_service_pb2.CancelTaskResponse()
            response.success = True
            response.message = "Task cancelled"
            return response
            
        except Exception as e:
            logger.error(f"Error cancelling task: {e}", exc_info=True)
            response = backend_service_pb2.CancelTaskResponse()
            response.success = False
            response.message = str(e)
            return response


# Placeholder classes if protobuf not generated
if 'backend_service_pb2' not in globals():
    class StartTaskRequest:
        def __init__(self):
            self.task_prompt = ""
            self.max_steps = 100
            self.user_id = ""
            self.browser_name = "firefox"
            self.browser_port = 9999
    
    class StartTaskResponse:
        def __init__(self):
            self.success = False
            self.task_id = ""
            self.message = ""
    
    class GetTaskStatusRequest:
        def __init__(self):
            self.task_id = ""
    
    class GetTaskStatusResponse:
        def __init__(self):
            self.success = False
            self.status = ""
            self.message = ""
    
    class CancelTaskRequest:
        def __init__(self):
            self.task_id = ""
    
    class CancelTaskResponse:
        def __init__(self):
            self.success = False
            self.message = ""


def serve(port: int = 50050):
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add servicer
    servicer = BackendServiceServicer()
    try:
        backend_service_pb2_grpc.add_BackendServiceServicer_to_server(servicer, server)
    except NameError:
        logger.warning("Using placeholder servicer (protobuf files not generated)")
    
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    server.start()
    
    logger.info(f"Backend Service gRPC server started on {listen_addr}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down Backend Service...")
        server.stop(grace=5)


if __name__ == "__main__":
    port = int(os.getenv("BACKEND_SERVICE_PORT", "50050"))
    serve(port)

