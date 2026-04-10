import asyncio
import logging

import uvicorn

from server.config import settings
from server.signaling import create_app


def configure_logging():
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence the extremely verbose ICE candidate pair logging
    logging.getLogger("aioice").setLevel(logging.WARNING)


async def main():
    configure_logging()
    app = create_app()
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
