import asyncio
from app.ai.core.wbs_engine import wbs_engine
from app.config import settings

def test_streaming():
    # Force mock for fast unit testing
    settings.USE_MOCK_LLM = True
    
    text = "Please build a simple web app with an auth page."
    
    async def run_test():
        print("Testing stream generation...")
        async for chunk in wbs_engine.stream_wbs_generation(text):
            print(chunk, end="")
        print("\n\nDone testing streaming.")
        
    asyncio.run(run_test())

if __name__ == "__main__":
    asyncio.run(test_streaming())
