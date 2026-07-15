from typing import Callable, Any
from app.models.task import Task

async def run_in_transaction(func: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Runs a series of database operations inside a single atomic transaction.
    The func callable should accept 'session' as a keyword argument or as its first argument,
    which will contain the active client session.
    """
    collection = Task.get_motor_inherit().io_bind
    client = collection.database.client
    
    async with await client.start_session() as session:
        async with session.start_transaction():
            return await func(session, *args, **kwargs)
