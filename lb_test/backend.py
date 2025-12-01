import os
import asyncio
from aiohttp import web

async def handle(request):
    server_id = os.environ.get('SERVER_ID', 'Unknown')
    # Simulate a small processing delay (optional, good for testing 'least_conn')
    await asyncio.sleep(0.05) 
    return web.Response(text=f"Server_{server_id}")

app = web.Application()
app.add_routes([web.get('/', handle)])

if __name__ == '__main__':
    web.run_app(app, port=8080)