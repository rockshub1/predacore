import asyncio

import httpx


async def test():
    proxy_url = "http://localhost:8080"
    endpoint = f"{proxy_url.rstrip('/')}/v1/messages"
    request_body = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1024,
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "dummy"
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(endpoint, json=request_body, headers=headers)
            print("Status:", resp.status_code)
            print("Body:", resp.text)
            resp.raise_for_status()
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
