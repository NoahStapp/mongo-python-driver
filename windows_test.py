import asyncio
import threading
from pymongo import AsyncMongoClient

clients = []


async def create_socket():
    c = AsyncMongoClient()
    await c.aconnect()

    clients.append(c)


async def use_socket():
    c = clients.pop()
    print(await c.db.command("hello"))


t1 = threading.Thread()
t2 = threading.Thread()
t1.target = asyncio.run(create_socket())
t2.target = asyncio.run(use_socket())

t1.run()
t2.run()
