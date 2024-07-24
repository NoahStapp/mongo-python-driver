import asyncio
import threading
from pymongo import AsyncMongoClient

clients = []


async def create_socket():
    c = AsyncMongoClient()
    print(f"{threading.current_thread().name}: {await c.db.command('hello')}")

    clients.append(c)


async def use_socket():
    while len(clients) < 1:
        pass
    c = clients.pop()
    print(f"{threading.current_thread().name}: {await c.db.command('hello')}")


def wrapper(func):
    asyncio.run(func())


t1 = threading.Thread(target=wrapper, args=(create_socket,))
t2 = threading.Thread(target=wrapper, args=(use_socket,))

t1.start()
t2.start()
