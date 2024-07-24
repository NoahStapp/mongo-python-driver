import asyncio
import threading
import socket

socks = []


async def create_socket():
    s = socket.socket()
    s.connect(("www.python.org", 80))
    s.settimeout(0.0)
    loop = asyncio.get_event_loop()
    print(f"{threading.current_thread().name}: {await asyncio.wait_for(loop.sock_sendall(s, bytes('hello', 'utf-8')), timeout=5)}")

    socks.append(s)


async def use_socket():
    while len(socks) < 1:
        pass
    s = socks.pop()
    loop = asyncio.get_event_loop()
    print(f"{threading.current_thread().name}: {await asyncio.wait_for(loop.sock_sendall(s, bytes('hello', 'utf-8')), timeout=5)}")


def wrapper(func):
    asyncio.run(func())


t1 = threading.Thread(target=wrapper, args=(create_socket,))
t2 = threading.Thread(target=wrapper, args=(use_socket,))

t1.start()
t2.start()
