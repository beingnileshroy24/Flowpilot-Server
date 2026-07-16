#!/usr/bin/env python3
import os
import sys
import asyncio
import logging

# Set PYTHONPATH so python can resolve 'app' imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import init_db
from app.services.sync_service import rebuild_vector_index

# Setup logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rebuild_script")

async def main():
    logger.info("Initializing connection to MongoDB...")
    try:
        await init_db()
        logger.info("MongoDB database connection established.")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        sys.exit(1)
        
    logger.info("Starting complete vector database reconstruction...")
    await rebuild_vector_index()
    logger.info("Vector database reconstruction process finished successfully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Rebuild script execution interrupted by user.")
        sys.exit(1)
