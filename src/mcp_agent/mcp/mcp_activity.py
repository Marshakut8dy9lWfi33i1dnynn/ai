# import functools
# from temporalio import activity
# from typing import Dict, Any, List, Callable, Awaitable
# from .gen_client import gen_client


# def mcp_activity(server_name: str, mcp_call: Callable):
#     def decorator(func):
#         @activity.defn
#         @functools.wraps(func)
#         async def wrapper(*activity_args, **activity_kwargs):
#             params = await func(*activity_args, **activity_kwargs)
#             async with gen_client(server_name) as client:
#                 return await mcp_call(client, params)

#         return wrapper

#     return decorator
