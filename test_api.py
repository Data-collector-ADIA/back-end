"""
Test script for the Browser Automation API
"""
import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_streaming_task():
    """Test the streaming task endpoint"""
    print("Testing streaming task endpoint...")
    
    payload = {
        "prompt": "Go to duckduckgo.com and search for 'weather'",
        "max_steps": 3
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("\nStreaming response:")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{API_URL}/api/v1/run",
            json=payload,
            stream=True,
            timeout=120
        )
        
        for line in response.iter_lines():
            if line:
                # Remove "data: " prefix
                json_str = line.decode('utf-8')
                if json_str.startswith('data: '):
                    json_str = json_str[6:]
                    
                data = json.loads(json_str)
                
                # Pretty print based on event type
                if data['type'] == 'task_start':
                    print(f"\n🚀 Task Started")
                    print(f"   Task: {data['task']}")
                    print(f"   Max Steps: {data['max_steps']}\n")
                    
                elif data['type'] == 'step':
                    print(f"\n📍 Step {data['step_number']}")
                    print(f"   URL: {data.get('url', 'N/A')}")
                    print(f"   Thinking: {data.get('thinking', 'N/A')[:100]}...")
                    print(f"   Next Goal: {data.get('next_goal', 'N/A')}")
                    print(f"   Actions: {len(data.get('actions', []))} action(s)")
                    
                elif data['type'] == 'task_complete':
                    print(f"\n✅ Task Complete")
                    print(f"   Success: {data['is_successful']}")
                    print(f"   Total Steps: {data['total_steps']}")
                    print(f"   Final Result: {data.get('final_result', 'N/A')}")
                    print(f"   URLs Visited: {', '.join(data.get('urls_visited', []))}")
                    
                elif data['type'] == 'error':
                    print(f"\n❌ Error")
                    print(f"   Type: {data['error_type']}")
                    print(f"   Message: {data['error']}")
        
        print("\n" + "=" * 60)
        print("Stream complete!")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON decode error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("="  * 60)
    print("Browser Automation API - Test Script")
    print("=" * 60)
    print()
    
    # Test health endpoint
    try:
        test_health()
    except Exception as e:
        print(f"⚠️  Health check failed: {e}")
        print("⚠️  Make sure the API server is running on port 8000")
        print("⚠️  Start it with: python api_server.py")
        exit(1)
    
    # Test streaming endpoint
    print("\nNote: This will start a real browser automation task.")
    print("Make sure all services are running (Browser service, etc.)")
    input("Press Enter to continue or Ctrl+C to cancel...")
    print()
    
    test_streaming_task()
