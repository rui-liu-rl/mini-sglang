# =====================================================================================
# FILE MAP -- utils/mp.py  ("mp" = message passing)
# =====================================================================================
# WHAT THIS FILE IS: small wrappers around ZeroMQ sockets: the Kitchen's PHONE LINES.
#
# ELI5: Rooms in the Kitchen are separate processes; they cannot just call each other's
#       functions. They talk through "phone lines" with a very simple rule:
#           PUSH/PULL : a one-way conveyor belt. One side puts parcels on, the other takes them
#                       off in order. If nobody is ready, parcels wait on the belt.
#           PUB/SUB   : a loudspeaker. One side announces, EVERY listener hears the same thing.
#       Each wrapper also packs the parcel (msgpack) before sending and unpacks it after.
#
# WHO USES IT: server/api_server.py (async), tokenizer/server.py, scheduler/io.py.
# WHAT IT CALLS: zmq (ZeroMQ) and msgpack.
#
# THE ONE IDEA TO REMEMBER ABOUT `create`:
#   A ZeroMQ line has two ends. One end must BIND (own the address, create=True); the other
#   CONNECTS (dials in, create=False). Which end binds is a design choice; here it is whoever
#   exists "permanently" (for example the scheduler binds the line it receives requests on).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")


# ELI5: the "put parcel on the belt" end (blocking-style, used from ordinary code).
class ZmqPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        # WHAT: bind if we own the address, otherwise connect to the owner.
        self.socket.bind(addr) if create else self.socket.connect(addr)
        # WHAT: encoder turns a message object into a plain dict (message/utils.py).
        self.encoder = encoder

    # WHAT: object -> dict -> msgpack bytes -> send. copy=False avoids an extra memory copy.
    # GOTCHA: ZeroMQ queues the parcel internally, so put() usually returns at once even if the
    #         receiver is busy or not even started yet.
    def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


# ELI5: the same belt, but for code that runs inside an asyncio event loop (the API server).
#       `await put(...)` lets other requests run while this send completes.
class ZmqAsyncPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


# ELI5: the "take parcel off the belt" end.
class ZmqPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    # WHAT: wait for the next parcel (this BLOCKS the process until one arrives), unpack it,
    #       rebuild the message object.
    def get(self) -> T:
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    # WHAT: receive the raw bytes WITHOUT unpacking.
    # WHY: scheduler rank 0 wants to forward the exact same bytes to the other GPU workers.
    #      Re-forwarding raw bytes skips a pointless decode and re-encode.
    def get_raw(self) -> bytes:
        return self.socket.recv()

    # WHAT: unpack bytes obtained earlier with get_raw.
    def decode(self, raw: bytes) -> T:
        return self.decoder(msgpack.unpackb(raw, raw=False))

    # ELI5: "Is anything waiting on the belt right now? Don't wait, just peek."
    # WHAT: poll with timeout=0 returns immediately; 0 events means the queue is empty.
    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()


# ELI5: the take-off end for asyncio code (the API server listens with this).
class ZmqAsyncPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    # WHAT: while this awaits, the event loop is free to serve other HTTP requests.
    async def get(self) -> T:
        event = await self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def stop(self):
        self.socket.close()
        self.context.term()


# ELI5: the LOUDSPEAKER. Scheduler rank 0 announces each new request to all the other GPU
#       workers so they all see exactly the same orders.
class ZmqPubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    # WHAT: broadcast bytes that are ALREADY packed (see ZmqPullQueue.get_raw).
    def put_raw(self, raw: bytes):
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


# ELI5: a LISTENER on the loudspeaker line.
class ZmqSubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        # WHAT: a SUB socket hears nothing until it subscribes. "" means "subscribe to everything".
        # GOTCHA: ZeroMQ PUB/SUB drops anything announced BEFORE the listener has connected (the
        #         "slow joiner" problem). The startup barrier in server/launch.py runs before any
        #         traffic exists, which is what makes this safe here.
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder

    def get(self) -> T:
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()
