import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Any

from browser_use import Agent, ChatGoogle, Browser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _suppress_logging():
	"""Suppress all logging output to prevent printing"""
	# Set logging level to CRITICAL to suppress all output
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


def _format_step_data(agent: Agent, step_number: int) -> dict[str, Any]:
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


async def run_agent_stream(prompt: str, max_steps: int = 100) -> AsyncGenerator[dict[str, Any], None]:
	"""
	Run an agent with the given prompt and stream updates as JSON.
	Uses a queue-based approach to capture updates in real-time.
	
	Args:
		prompt: The task prompt for the agent
		max_steps: Maximum number of steps the agent can take
		
	Yields:
		dict: JSON-serializable dictionaries containing step updates
	"""
	# Suppress all logging output
	_suppress_logging()
	
	# Initialize agent with Gemini
	# GOOGLE_API_KEY should be set in environment or .env file
	llm = ChatGoogle(model='gemini-flash-latest')
	browser = Browser(headless=True)
	agent = Agent(
		task=prompt,
		llm=llm,
		browser=browser,
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
	
	try:
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
		raise
	finally:
		# Cleanup browser
		if browser:
			try:
				await browser.stop()
			except Exception:
				pass


# Main function for testing
async def main():
	"""Example usage of the streaming function"""
	prompt = "Search for 'browser automation' on DuckDuckGo"
	
	print("Starting agent stream...")
	async for update in run_agent_stream(prompt, max_steps=5 ):
		# Print JSON output
		print(json.dumps(update, indent=2, default=str))


if __name__ == "__main__":
	asyncio.run(main())

