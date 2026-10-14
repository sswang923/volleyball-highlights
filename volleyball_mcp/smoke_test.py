#!/usr/bin/env python3
"""Launch the MCP server over stdio and exercise its tools (no Claude needed).

    cd volleyball_mcp && UV_PROJECT_ENVIRONMENT=~/.venvs/volleyball-mcp uv run smoke_test.py [game]
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

GAME = sys.argv[1] if len(sys.argv) > 1 else "20261004_game02"


async def main():
    params = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).with_name("server.py"))])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("tools:", [t.name for t in tools.tools])

            async def call(name, **kw):
                res = await s.call_tool(name, kw)
                text = res.content[0].text if res.content else ""
                print(f"\n== {name}({kw})  error={res.isError}  {len(text)} chars")
                print(text[:900])
                return json.loads(text) if text.startswith("{") else text

            await call("list_games")
            job = await call("detect_rallies", game=GAME, set_scores="19-25,25-18")
            for _ in range(60):
                st = await call("job_status", job_id=job["job_id"])
                if st["state"] != "running":
                    break
                await asyncio.sleep(2)
            await call("rally_summary", game=GAME, set_number=2, limit=5)
            await call("evaluate_game", game=GAME)
            await call("detect_rallies", game="bad name")


asyncio.run(main())
