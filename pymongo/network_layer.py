# Copyright 2015-present MongoDB, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Internal network layer helper methods."""
from __future__ import annotations

import asyncio
import collections
import errno
import socket
import struct
import sys
import time
from asyncio import AbstractEventLoop, BaseTransport, BufferedProtocol, Future, Transport
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
)

from pymongo import _csot, ssl_support
from pymongo._asyncio_task import create_task
from pymongo.common import MAX_MESSAGE_SIZE
from pymongo.compression_support import decompress
from pymongo.errors import ProtocolError, _OperationCancelled
from pymongo.message import _UNPACK_REPLY, _OpMsg, _OpReply
from pymongo.socket_checker import _errno_from_exception

try:
    from ssl import SSLError, SSLSocket

    _HAVE_SSL = True
except ImportError:
    _HAVE_SSL = False

try:
    from pymongo.pyopenssl_context import (
        BLOCKING_IO_LOOKUP_ERROR,
        BLOCKING_IO_READ_ERROR,
        BLOCKING_IO_WRITE_ERROR,
        _sslConn,
    )

    _HAVE_PYOPENSSL = True
except ImportError:
    _HAVE_PYOPENSSL = False
    _sslConn = SSLSocket  # type: ignore
    from pymongo.ssl_support import (  # type: ignore[assignment]
        BLOCKING_IO_LOOKUP_ERROR,
        BLOCKING_IO_READ_ERROR,
        BLOCKING_IO_WRITE_ERROR,
    )

if TYPE_CHECKING:
    from pymongo.asynchronous.pool import AsyncConnection
    from pymongo.synchronous.pool import Connection

_UNPACK_HEADER = struct.Struct("<iiii").unpack
_UNPACK_COMPRESSION_HEADER = struct.Struct("<iiB").unpack
_POLL_TIMEOUT = 0.5
# Errors raised by sockets (and TLS sockets) when in non-blocking mode.
BLOCKING_IO_ERRORS = (BlockingIOError, BLOCKING_IO_LOOKUP_ERROR, *ssl_support.BLOCKING_IO_ERRORS)


# These socket-based I/O methods are for KMS requests and any other network operations that do not use
# the MongoDB wire protocol
async def async_socket_sendall(sock: Union[socket.socket, _sslConn], buf: bytes) -> None:
    timeout = sock.gettimeout()
    sock.settimeout(0.0)
    loop = asyncio.get_running_loop()
    try:
        if _HAVE_SSL and isinstance(sock, (SSLSocket, _sslConn)):
            await asyncio.wait_for(_async_socket_sendall_ssl(sock, buf, loop), timeout=timeout)
        else:
            await asyncio.wait_for(loop.sock_sendall(sock, buf), timeout=timeout)  # type: ignore[arg-type]
    except asyncio.TimeoutError as exc:
        # Convert the asyncio.wait_for timeout error to socket.timeout which pool.py understands.
        raise socket.timeout("timed out") from exc
    finally:
        sock.settimeout(timeout)


if sys.platform != "win32":

    async def _async_socket_sendall_ssl(
        sock: Union[socket.socket, _sslConn], buf: bytes, loop: AbstractEventLoop
    ) -> None:
        view = memoryview(buf)
        sent = 0

        def _is_ready(fut: Future) -> None:
            if fut.done():
                return
            fut.set_result(None)

        while sent < len(buf):
            try:
                sent += sock.send(view[sent:])
            except BLOCKING_IO_ERRORS as exc:
                fd = sock.fileno()
                # Check for closed socket.
                if fd == -1:
                    raise SSLError("Underlying socket has been closed") from None
                if isinstance(exc, BLOCKING_IO_READ_ERROR):
                    fut = loop.create_future()
                    loop.add_reader(fd, _is_ready, fut)
                    try:
                        await fut
                    finally:
                        loop.remove_reader(fd)
                if isinstance(exc, BLOCKING_IO_WRITE_ERROR):
                    fut = loop.create_future()
                    loop.add_writer(fd, _is_ready, fut)
                    try:
                        await fut
                    finally:
                        loop.remove_writer(fd)
                if _HAVE_PYOPENSSL and isinstance(exc, BLOCKING_IO_LOOKUP_ERROR):
                    fut = loop.create_future()
                    loop.add_reader(fd, _is_ready, fut)
                    try:
                        loop.add_writer(fd, _is_ready, fut)
                        await fut
                    finally:
                        loop.remove_reader(fd)
                        loop.remove_writer(fd)

    async def _async_socket_receive_ssl(
        conn: _sslConn, length: int, loop: AbstractEventLoop, once: Optional[bool] = False
    ) -> memoryview:
        mv = memoryview(bytearray(length))
        total_read = 0

        def _is_ready(fut: Future) -> None:
            if fut.done():
                return
            fut.set_result(None)

        while total_read < length:
            try:
                read = conn.recv_into(mv[total_read:])
                if read == 0:
                    raise OSError("connection closed")
                # KMS responses update their expected size after the first batch, stop reading after one loop
                if once:
                    return mv[:read]
                total_read += read
            except BLOCKING_IO_ERRORS as exc:
                fd = conn.fileno()
                # Check for closed socket.
                if fd == -1:
                    raise SSLError("Underlying socket has been closed") from None
                if isinstance(exc, BLOCKING_IO_READ_ERROR):
                    fut = loop.create_future()
                    loop.add_reader(fd, _is_ready, fut)
                    try:
                        await fut
                    finally:
                        loop.remove_reader(fd)
                if isinstance(exc, BLOCKING_IO_WRITE_ERROR):
                    fut = loop.create_future()
                    loop.add_writer(fd, _is_ready, fut)
                    try:
                        await fut
                    finally:
                        loop.remove_writer(fd)
                if _HAVE_PYOPENSSL and isinstance(exc, BLOCKING_IO_LOOKUP_ERROR):
                    fut = loop.create_future()
                    loop.add_reader(fd, _is_ready, fut)
                    try:
                        loop.add_writer(fd, _is_ready, fut)
                        await fut
                    finally:
                        loop.remove_reader(fd)
                        loop.remove_writer(fd)
        return mv

else:
    # The default Windows asyncio event loop does not support loop.add_reader/add_writer:
    # https://docs.python.org/3/library/asyncio-platforms.html#asyncio-platform-support
    # Note: In PYTHON-4493 we plan to replace this code with asyncio streams.
    async def _async_socket_sendall_ssl(
        sock: Union[socket.socket, _sslConn], buf: bytes, dummy: AbstractEventLoop
    ) -> None:
        view = memoryview(buf)
        total_length = len(buf)
        total_sent = 0
        # Backoff starts at 1ms, doubles on timeout up to 512ms, and halves on success
        # down to 1ms.
        backoff = 0.001
        while total_sent < total_length:
            try:
                sent = sock.send(view[total_sent:])
            except BLOCKING_IO_ERRORS:
                await asyncio.sleep(backoff)
                sent = 0
            if sent > 0:
                backoff = max(backoff / 2, 0.001)
            else:
                backoff = min(backoff * 2, 0.512)
            total_sent += sent

    async def _async_socket_receive_ssl(
        conn: _sslConn, length: int, dummy: AbstractEventLoop, once: Optional[bool] = False
    ) -> memoryview:
        mv = memoryview(bytearray(length))
        total_read = 0
        # Backoff starts at 1ms, doubles on timeout up to 512ms, and halves on success
        # down to 1ms.
        backoff = 0.001
        while total_read < length:
            try:
                read = conn.recv_into(mv[total_read:])
                if read == 0:
                    raise OSError("connection closed")
                # KMS responses update their expected size after the first batch, stop reading after one loop
                if once:
                    return mv[:read]
            except BLOCKING_IO_ERRORS:
                await asyncio.sleep(backoff)
                read = 0
            if read > 0:
                backoff = max(backoff / 2, 0.001)
            else:
                backoff = min(backoff * 2, 0.512)
            total_read += read
        return mv


def sendall(sock: Union[socket.socket, _sslConn], buf: bytes) -> None:
    sock.sendall(buf)


async def _poll_cancellation(conn: AsyncConnection) -> None:
    while True:
        if conn.cancel_context.cancelled:
            return

        await asyncio.sleep(_POLL_TIMEOUT)


async def async_receive_data_socket(
    sock: Union[socket.socket, _sslConn], length: int
) -> memoryview:
    sock_timeout = sock.gettimeout()
    timeout = sock_timeout

    sock.settimeout(0.0)
    loop = asyncio.get_running_loop()
    try:
        if _HAVE_SSL and isinstance(sock, (SSLSocket, _sslConn)):
            return await asyncio.wait_for(
                _async_socket_receive_ssl(sock, length, loop, once=True),  # type: ignore[arg-type]
                timeout=timeout,
            )
        else:
            return await asyncio.wait_for(
                _async_socket_receive(sock, length, loop),  # type: ignore[arg-type]
                timeout=timeout,
            )
    except asyncio.TimeoutError as err:
        raise socket.timeout("timed out") from err
    finally:
        sock.settimeout(sock_timeout)


async def _async_socket_receive(
    conn: socket.socket, length: int, loop: AbstractEventLoop
) -> memoryview:
    mv = memoryview(bytearray(length))
    bytes_read = 0
    while bytes_read < length:
        chunk_length = await loop.sock_recv_into(conn, mv[bytes_read:])
        if chunk_length == 0:
            raise OSError("connection closed")
        bytes_read += chunk_length
    return mv


_PYPY = "PyPy" in sys.version


def wait_for_read(conn: Connection, deadline: Optional[float]) -> None:
    """Block until at least one byte is read, or a timeout, or a cancel."""
    sock = conn.conn.sock
    timed_out = False
    # Check if the connection's socket has been manually closed
    if sock.fileno() == -1:
        return
    while True:
        # SSLSocket can have buffered data which won't be caught by select.
        if hasattr(sock, "pending") and sock.pending() > 0:
            readable = True
        else:
            # Wait up to 500ms for the socket to become readable and then
            # check for cancellation.
            if deadline:
                remaining = deadline - time.monotonic()
                # When the timeout has expired perform one final check to
                # see if the socket is readable. This helps avoid spurious
                # timeouts on AWS Lambda and other FaaS environments.
                if remaining <= 0:
                    timed_out = True
                timeout = max(min(remaining, _POLL_TIMEOUT), 0)
            else:
                timeout = _POLL_TIMEOUT
            readable = conn.socket_checker.select(sock, read=True, timeout=timeout)
        if conn.cancel_context.cancelled:
            raise _OperationCancelled("operation cancelled")
        if readable:
            return
        if timed_out:
            raise socket.timeout("timed out")


def receive_data(conn: Connection, length: int, deadline: Optional[float]) -> memoryview:
    buf = bytearray(length)
    mv = memoryview(buf)
    bytes_read = 0
    # To support cancelling a network read, we shorten the socket timeout and
    # check for the cancellation signal after each timeout. Alternatively we
    # could close the socket but that does not reliably cancel recv() calls
    # on all OSes.
    # When the timeout has expired we perform one final non-blocking recv.
    # This helps avoid spurious timeouts when the response is actually already
    # buffered on the client.
    orig_timeout = conn.conn.gettimeout()
    try:
        while bytes_read < length:
            try:
                # Use the legacy wait_for_read cancellation approach on PyPy due to PYTHON-5011.
                if _PYPY:
                    wait_for_read(conn, deadline)
                    if _csot.get_timeout() and deadline is not None:
                        conn.set_conn_timeout(max(deadline - time.monotonic(), 0))
                else:
                    if deadline is not None:
                        short_timeout = min(max(deadline - time.monotonic(), 0), _POLL_TIMEOUT)
                    else:
                        short_timeout = _POLL_TIMEOUT
                    conn.set_conn_timeout(short_timeout)

                chunk_length = conn.conn.recv_into(mv[bytes_read:])
            except BLOCKING_IO_ERRORS:
                if conn.cancel_context.cancelled:
                    raise _OperationCancelled("operation cancelled") from None
                # We reached the true deadline.
                raise socket.timeout("timed out") from None
            except socket.timeout:
                if conn.cancel_context.cancelled:
                    raise _OperationCancelled("operation cancelled") from None
                if _PYPY:
                    # We reached the true deadline.
                    raise
                continue
            except OSError as exc:
                if conn.cancel_context.cancelled:
                    raise _OperationCancelled("operation cancelled") from None
                if _errno_from_exception(exc) == errno.EINTR:
                    continue
                raise
            if chunk_length == 0:
                raise OSError("connection closed")

            bytes_read += chunk_length
    finally:
        conn.set_conn_timeout(orig_timeout)

    return mv


class NetworkingInterfaceBase:
    def __init__(self, conn: Any):
        self.conn = conn

    @property
    def gettimeout(self) -> Any:
        raise NotImplementedError

    def settimeout(self, timeout: float | None) -> None:
        raise NotImplementedError

    def close(self) -> Any:
        raise NotImplementedError

    def is_closing(self) -> bool:
        raise NotImplementedError

    @property
    def get_conn(self) -> Any:
        raise NotImplementedError

    @property
    def sock(self) -> Any:
        raise NotImplementedError


class AsyncNetworkingInterface(NetworkingInterfaceBase):
    def __init__(self, conn: tuple[Transport, PyMongoProtocol]):
        super().__init__(conn)

    @property
    def gettimeout(self) -> float | None:
        return self.conn[1].gettimeout

    def settimeout(self, timeout: float | None) -> None:
        self.conn[1].settimeout(timeout)

    async def close(self) -> None:
        self.conn[0].abort()
        await self.conn[1].wait_closed()

    def is_closing(self) -> bool:
        return self.conn[0].is_closing()

    @property
    def get_conn(self) -> PyMongoProtocol:
        return self.conn[1]

    @property
    def sock(self) -> socket.socket:
        return self.conn[0].get_extra_info("socket")


class NetworkingInterface(NetworkingInterfaceBase):
    def __init__(self, conn: Union[socket.socket, _sslConn]):
        super().__init__(conn)

    def gettimeout(self) -> float | None:
        return self.conn.gettimeout()

    def settimeout(self, timeout: float | None) -> None:
        self.conn.settimeout(timeout)

    def close(self) -> None:
        self.conn.close()

    def is_closing(self) -> bool:
        return self.conn.is_closing()

    @property
    def get_conn(self) -> Union[socket.socket, _sslConn]:
        return self.conn

    @property
    def sock(self) -> Union[socket.socket, _sslConn]:
        return self.conn

    def fileno(self) -> int:
        return self.conn.fileno()

    def recv_into(self, buffer: bytes) -> int:
        return self.conn.recv_into(buffer)


class PyMongoProtocol(BufferedProtocol):
    def __init__(self, timeout: Optional[float] = None, buffer_size: int = 2**14):
        self._buffer_size = buffer_size
        self.transport: Transport = None  # type: ignore[assignment]
        self._buffer = memoryview(bytearray(self._buffer_size))
        self._overflow: Optional[memoryview] = None
        self._start_index = 0
        self._end_index = 0
        self._overflow_index = 0
        self._body_size = 0
        self._op_code: int = None  # type: ignore[assignment]
        self._connection_lost = False
        self._paused = False
        self._drain_waiter: Optional[Future] = None
        self._read_waiter: Optional[Future] = None
        self._timeout = timeout
        self._is_compressed = False
        self._compressor_id: Optional[int] = None
        self._need_compression_header = False
        self._max_message_size = MAX_MESSAGE_SIZE
        self._request_id: Optional[int] = None
        self._closed = asyncio.get_running_loop().create_future()
        self._expecting_header = True
        self._pending_messages: collections.deque[Future] = collections.deque()
        self._done_messages: collections.deque[Future] = collections.deque()

    def settimeout(self, timeout: float | None) -> None:
        self._timeout = timeout

    @property
    def gettimeout(self) -> float | None:
        """The configured timeout for the socket that underlies our protocol pair."""
        return self._timeout

    def connection_made(self, transport: BaseTransport) -> None:
        """Called exactly once when a connection is made.
        The transport argument is the transport representing the write side of the connection.
        """
        self.transport = transport  # type: ignore[assignment]

    async def write(self, message: bytes) -> None:
        """Write a message to this connection's transport."""
        if self.transport.is_closing():
            raise OSError("Connection is closed")
        self.transport.write(message)
        await self._drain_helper()
        self.transport.resume_reading()

    async def read(self, request_id: Optional[int], max_message_size: int) -> tuple[bytes, int]:
        """Read a single MongoDB Wire Protocol message from this connection."""
        if self.transport:
            self.transport.resume_reading()
        if self._done_messages:
            message = await self._done_messages.popleft()
        else:
            self._expecting_header = True
            self._max_message_size = max_message_size
            self._request_id = request_id
            self._end_index = 0
            self._overflow_index = 0
            self._body_size = 0
            self._start_index = 0
            self._op_code = None  # type: ignore[assignment]
            self._overflow = None
            if self.transport and self.transport.is_closing():
                raise OSError("Connection is closed")
            read_waiter = asyncio.get_running_loop().create_future()
            self._pending_messages.append(read_waiter)
            try:
                message = await read_waiter
            finally:
                if read_waiter in self._done_messages:
                    self._done_messages.remove(read_waiter)
        if message:
            start, end, op_code, overflow, overflow_index = (
                message[0],
                message[1],
                message[2],
                message[3],
                message[4],
            )
            if self._is_compressed:
                header_size = 25
            else:
                header_size = 16
            if overflow is not None:
                if self._is_compressed and self._compressor_id is not None:
                    return decompress(
                        memoryview(
                            bytearray(self._buffer[start + header_size : self._end_index])
                            + bytearray(overflow[:overflow_index])
                        ),
                        self._compressor_id,
                    ), op_code
                else:
                    return memoryview(
                        bytearray(self._buffer[start + header_size : self._end_index])
                        + bytearray(overflow[:overflow_index])
                    ), op_code
            else:
                if self._is_compressed and self._compressor_id is not None:
                    return decompress(
                        memoryview(self._buffer[start + header_size : end]),
                        self._compressor_id,
                    ), op_code
                else:
                    return memoryview(self._buffer[start + header_size : end]), op_code
        raise OSError("connection closed")

    def get_buffer(self, sizehint: int) -> memoryview:
        """Called to allocate a new receive buffer."""
        if self._overflow is not None:
            return self._overflow[self._overflow_index :]
        return self._buffer[self._end_index :]

    def buffer_updated(self, nbytes: int) -> None:
        """Called when the buffer was updated with the received data"""
        # Wrote 0 bytes into a non-empty buffer, signal connection closed
        if nbytes == 0:
            self.connection_lost(OSError("connection closed"))
            return
        else:
            # Wrote data into overflow buffer
            if self._overflow is not None:
                self._overflow_index += nbytes
            # Wrote data into default buffer
            else:
                if self._expecting_header:
                    try:
                        self._body_size, self._op_code = self.process_header()
                        self._request_id = None
                    except ProtocolError as exc:
                        self.connection_lost(exc)
                        return
                    self._expecting_header = False
                    # The new message's data is too large for the default buffer, allocate a new buffer for this message
                    if self._body_size > self._buffer_size - (self._end_index + nbytes):
                        self._overflow = memoryview(bytearray(self._body_size))
                self._end_index += nbytes
            # All data of the current message has been received
            if self._end_index + self._overflow_index - self._start_index >= self._body_size:
                if self._pending_messages:
                    result = self._pending_messages.popleft()
                else:
                    result = asyncio.get_running_loop().create_future()
                # Future has been cancelled, close this connection
                if result.done():
                    self.connection_lost(None)
                    return
                # Necessary values to construct message from buffers
                result.set_result(
                    (
                        self._start_index,
                        self._body_size + self._start_index,
                        self._op_code,
                        self._overflow,
                        self._overflow_index,
                    )
                )
                # If the current message has an overflow buffer, then the entire default buffer is full
                if self._overflow:
                    self._start_index = self._end_index
                # Update the buffer's first written offset to reflect this message's size
                else:
                    self._start_index += self._body_size
                self._done_messages.append(result)
                # If at least one header's worth of data remains after the current message, reprocess all leftover data
                if self._end_index - self._start_index >= 16:
                    self._read_waiter = asyncio.get_running_loop().create_future()
                    self._pending_messages.append(self._read_waiter)
                    nbytes_reprocess = self._end_index - self._start_index
                    self._end_index -= nbytes_reprocess
                    # Reset internal state to expect a new message
                    self._expecting_header = True
                    self._body_size = 0
                    self._op_code = None  # type: ignore[assignment]
                    self._overflow = None
                    self._overflow_index = 0
                    self.buffer_updated(nbytes_reprocess)
                # Pause reading to avoid storing an arbitrary number of messages in memory before necessary
                self.transport.pause_reading()

    def process_header(self) -> tuple[int, int]:
        """Unpack a MongoDB Wire Protocol header."""
        length, _, response_to, op_code = _UNPACK_HEADER(
            self._buffer[self._start_index : self._start_index + 16]
        )
        # No request_id for exhaust cursor "getMore".
        if self._request_id is not None:
            if self._request_id != response_to:
                raise ProtocolError(
                    f"Got response id {response_to!r} but expected {self._request_id!r}"
                )
        if length <= 16:
            raise ProtocolError(
                f"Message length ({length!r}) not longer than standard message header size (16)"
            )
        if length > self._max_message_size:
            raise ProtocolError(
                f"Message length ({length!r}) is larger than server max "
                f"message size ({self._max_message_size!r})"
            )
        if op_code == 2012:
            self._is_compressed = True
            if self._end_index >= 25:
                op_code, _, self._compressor_id = _UNPACK_COMPRESSION_HEADER(
                    self._buffer[self._start_index + 16 : self._start_index + 25]
                )
            else:
                self._need_compression_header = True

        return length, op_code

    def pause_writing(self) -> None:
        assert not self._paused
        self._paused = True

    def resume_writing(self) -> None:
        assert self._paused
        self._paused = False

        if self._drain_waiter and not self._drain_waiter.done():
            self._drain_waiter.set_result(None)

    def connection_lost(self, exc: Exception | None) -> None:
        self._connection_lost = True
        pending = list(self._pending_messages)
        for msg in pending:
            if not msg.done():
                if exc is None:
                    msg.set_result(None)
                else:
                    msg.set_exception(exc)
            self._done_messages.append(msg)

        if not self._closed.done():
            if exc is None:
                self._closed.set_result(None)
            else:
                self._closed.set_exception(exc)

        # Wake up the writer(s) if currently paused.
        if not self._paused:
            return

        if self._drain_waiter and not self._drain_waiter.done():
            if exc is None:
                self._drain_waiter.set_result(None)
            else:
                self._drain_waiter.set_exception(exc)

    async def _drain_helper(self) -> None:
        if self._connection_lost:
            raise ConnectionResetError("Connection lost")
        if not self._paused:
            return
        self._drain_waiter = asyncio.get_running_loop().create_future()
        await self._drain_waiter

    def data(self) -> memoryview:
        return self._buffer

    async def wait_closed(self) -> None:
        await asyncio.wait([self._closed])


async def async_sendall(conn: PyMongoProtocol, buf: bytes) -> None:
    try:
        await asyncio.wait_for(conn.write(buf), timeout=conn.gettimeout)
    except asyncio.TimeoutError as exc:
        # Convert the asyncio.wait_for timeout error to socket.timeout which pool.py understands.
        raise socket.timeout("timed out") from exc


async def async_receive_message(
    conn: AsyncConnection,
    request_id: Optional[int],
    max_message_size: int = MAX_MESSAGE_SIZE,
) -> Union[_OpReply, _OpMsg]:
    """Receive a raw BSON message or raise socket.error."""
    timeout: Optional[Union[float, int]]
    timeout = conn.conn.gettimeout
    if _csot.get_timeout():
        deadline = _csot.get_deadline()
    else:
        if timeout:
            deadline = time.monotonic() + timeout
        else:
            deadline = None
    if deadline:
        # When the timeout has expired perform one final check to
        # see if the socket is readable. This helps avoid spurious
        # timeouts on AWS Lambda and other FaaS environments.
        timeout = max(deadline - time.monotonic(), 0)

    cancellation_task = create_task(_poll_cancellation(conn))
    read_task = create_task(conn.conn.get_conn.read(request_id, max_message_size))
    tasks = [read_task, cancellation_task]
    try:
        done, pending = await asyncio.wait(
            tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending)
        if len(done) == 0:
            raise socket.timeout("timed out")
        if read_task in done:
            data, op_code = read_task.result()
            try:
                unpack_reply = _UNPACK_REPLY[op_code]
            except KeyError:
                raise ProtocolError(
                    f"Got opcode {op_code!r} but expected {_UNPACK_REPLY.keys()!r}"
                ) from None
            return unpack_reply(data)
        raise _OperationCancelled("operation cancelled")
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.wait(tasks)
        raise


def receive_message(
    conn: Connection, request_id: Optional[int], max_message_size: int = MAX_MESSAGE_SIZE
) -> Union[_OpReply, _OpMsg]:
    """Receive a raw BSON message or raise socket.error."""
    if _csot.get_timeout():
        deadline = _csot.get_deadline()
    else:
        timeout = conn.conn.gettimeout()
        if timeout:
            deadline = time.monotonic() + timeout
        else:
            deadline = None
    # Ignore the response's request id.
    length, _, response_to, op_code = _UNPACK_HEADER(receive_data(conn, 16, deadline))
    # No request_id for exhaust cursor "getMore".
    if request_id is not None:
        if request_id != response_to:
            raise ProtocolError(f"Got response id {response_to!r} but expected {request_id!r}")
    if length <= 16:
        raise ProtocolError(
            f"Message length ({length!r}) not longer than standard message header size (16)"
        )
    if length > max_message_size:
        raise ProtocolError(
            f"Message length ({length!r}) is larger than server max "
            f"message size ({max_message_size!r})"
        )
    if op_code == 2012:
        op_code, _, compressor_id = _UNPACK_COMPRESSION_HEADER(receive_data(conn, 9, deadline))
        data = decompress(receive_data(conn, length - 25, deadline), compressor_id)
    else:
        data = receive_data(conn, length - 16, deadline)

    try:
        unpack_reply = _UNPACK_REPLY[op_code]
    except KeyError:
        raise ProtocolError(
            f"Got opcode {op_code!r} but expected {_UNPACK_REPLY.keys()!r}"
        ) from None
    return unpack_reply(data)
