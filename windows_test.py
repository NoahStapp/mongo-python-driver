import asyncio
import threading
import socket

socks = []


async def create_socket():
    s = socket.socket()
    s.connect(("www.python.org", 80))
    s.settimeout(0.0)
    print(f"{threading.current_thread().name}: {s.sendall(bytes('hello', 'utf-8'))}")

    socks.append(s)


async def use_socket():
    while len(socks) < 1:
        pass
    s = socks.pop()
    print(f"{threading.current_thread().name}: {s.sendall(bytes('hello', 'utf-8'))}")


def wrapper(func):
    asyncio.run(func())


t1 = threading.Thread(target=wrapper, args=(create_socket,))
t2 = threading.Thread(target=wrapper, args=(use_socket,))

t1.start()
t2.start()
