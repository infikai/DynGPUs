import asyncio
import os
import random
from aiohttp import web

async def handle(request):
    server_id = os.environ.get('SERVER_ID', 'Unknown')
    
    # 1. Simulate "Prefill" (Initial delay before first token)
    # vLLM takes longer to start if the prompt is long
    prefill_time = random.uniform(0.1, 0.5) 
    await asyncio.sleep(prefill_time)

    # 2. Simulate "Streaming" (Holding connection open)
    # Some requests are short (0.5s), some are long (3.0s)
    generation_time = random.uniform(0.5, 3.0) 
    
    response = web.StreamResponse()
    response.headers['Content-Type'] = 'text/plain'
    # Important: Tell Nginx this is a stream
    response.enable_chunked_encoding() 
    
    await response.prepare(request)
    
    # Stream "tokens"
    chunks = 5
    for i in range(chunks):
        await asyncio.sleep(generation_time / chunks)
        await response.write(f"Token_{i} ".encode('utf-8'))
        
    await response.write(f"[End from {server_id}]\n".encode('utf-8'))
    await response.write_eof()
    
    return response

app = web.Application()
app.add_routes([web.get('/', handle)])

if __name__ == '__main__':
    web.run_app(app, port=8080)