import asyncio
import threading
from pymongo import AsyncMongoClient
import socket

clients = []


async def create_socket():
    c = socket.socket()
    c.connect(("www.python.org", 80))
    c.settimeout(0.0)
    loop = asyncio.get_event_loop()
    print(f"{threading.current_thread().name}: {await asyncio.wait_for(loop.sock_sendall(c, bytes('hello', 'utf-8')), timeout=5)}")

    clients.append(c)


async def use_socket():
    while len(clients) < 1:
        pass
    c = clients.pop()
    c.settimeout(0.0)
    loop = asyncio.get_event_loop()
    print(f"{threading.current_thread().name}: {await asyncio.wait_for(loop.sock_sendall(c, bytes('hello', 'utf-8')), timeout=5)}")


def wrapper(func):
    asyncio.run(func())


t1 = threading.Thread(target=wrapper, args=(create_socket,))
t2 = threading.Thread(target=wrapper, args=(use_socket,))

t1.start()
t2.start()
