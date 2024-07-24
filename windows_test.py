import asyncio
import socket
import threading

socks = []


async def create_socket():
    sock = socket.socket()
    sock.connect(("www.python.org", 80))

    socks.append(sock)


async def use_socket():
    sock = socks.pop()
    timeout = sock.gettimeout()
    sock.settimeout(0.0)
    loop = asyncio.get_event_loop()
    try:
        await asyncio.wait_for(loop.sock_sendall(sock, bytes("hello", "utf8")), timeout=timeout)
    finally:
        sock.settimeout(timeout)


t1 = threading.Thread()
t2 = threading.Thread()
t1.target = asyncio.run(create_socket())
t2.target = asyncio.run(use_socket())

t1.run()
t2.run()
