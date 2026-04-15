# Project Development Context
## Current Task: Implement Core Strategic Engine
## Previous Tasks:
1. Created LLMPlanner class structure
2. Implemented API key environment variable handling
3. Added risk calculation module
## Current State:
- API keys must use os.getenv() pattern
- All modules must import from core.utils
- Current model: qwen/qwen3-235b-a22b
## Coding Standards:
1. All LLM clients must use environment variables
2. Modules must follow the pattern: 
   ```python
   from core.utils import get_api_key
   class StrategicPlanner:
       def __init__(self):
           self.api_key = get_api_key()