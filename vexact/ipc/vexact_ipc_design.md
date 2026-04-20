# VExact IPC Design

## Directory Structure

```
vexact/
├── ipc/
│   ├── __init__.py
│   ├── request_channel.py   # Request channel (async streaming)
│   ├── control_channel.py   # Control channel (sync RPC, one-to-many)
│   └── messages.py          # msgspec message definitions
├── request.py               # DriverRequest/Output with enc/dec hooks
├── driver_client.py         # DriverClient
├── driver_worker_proxy.py   # DriverWorkerProxy
└── worker_proxy.py          # WorkerProxy
```

## Two Communication Channels

| Channel | Direction                  | Mode                          | ZMQ Socket    | Protocol             | Purpose                        |
| ------- | -------------------------- | ----------------------------- | ------------- | -------------------- | ------------------------------ |
| Request | Client ↔ DriverWorkerProxy | Async bidirectional streaming | DEALER/ROUTER | `ipc://`             | submit request, receive output |
| Control | Client → All Proxies       | Sync RPC (one-to-many)        | REQ/REP       | `ipc://` or `tcp://` | sleep, wake_up                 |

### Request Channel (DEALER/ROUTER)

- **DEALER**: Async send/receive, can send multiple requests without blocking
- **ROUTER**: Tracks message origins, enables targeted replies to specific clients
- **IPC only**: For local inter-process communication

### Control Channel (REQ/REP)

- **REQ**: Request socket (client-side)
- **REP**: Reply socket (server-side)
- **Supports**: IPC, TCP (IPv4), TCP (IPv6) with automatic dual-stack

## Core Components

### 1. IPC Layer

#### messages.py

```python
# ipc/messages.py
class ControlRequest(msgspec.Struct):
    method: str
    args: tuple = ()
    kwargs: dict = {}

class ControlResponse(msgspec.Struct):
    success: bool
    result: Any = None
    error: str | None = None
```

#### request.py

```python
# request.py
class DriverRequest(msgspec.Struct):
    request_id: str
    generation_config: GenerationConfig
    input_ids_list: list[int]

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        """Encode non-msgspec types (for msgspec encoder)"""
        if isinstance(obj, GenerationConfig):
            return obj.to_dict()
        ...

    @staticmethod
    def dec_hook(type_: type, obj: Any) -> Any:
        """Decode non-msgspec types (for msgspec decoder)"""
        if type_ is GenerationConfig:
            return GenerationConfig.from_dict(obj)
        ...

class DriverRequestOutput(msgspec.Struct):
    request_id: str
    new_token_ids: list[int]
    new_logprobs: list[float] | None = None
```

#### request_channel.py

```python
# ipc/request_channel.py
class RequestChannelClient:
    """DEALER socket for async bidirectional communication"""
    def __init__(self, address: str):
        # Auto-starts on first submit()
        ...

    async def submit(self, request: DriverRequest) -> AsyncIterator[DriverRequestOutput]:
        # Auto-starts if not running
        ...

    def close(self):
        # Cleanup resources
        ...

class RequestChannelServer:
    """ROUTER socket, background thread handles requests and replies"""
    def __init__(self, address: str, on_request: Callable[[DriverRequest], None], poll_outputs: Callable):
        ...

    def start(self):
        # Start background thread
        ...

    def stop(self):
        # Stop background thread
        ...
```

#### control_channel.py

```python
# ipc/control_channel.py
class ControlChannelClient:
    """One-to-many, manages multiple REQ sockets internally"""
    def __init__(self, proxy_addresses: list[str]):
        # Supports ipc://, tcp:// (IPv4/IPv6)
        ...

    def collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        # Execute on all proxies
        ...

    def close(self):
        ...

class ControlChannelServer:
    """REP socket, background thread handles RPC"""
    def __init__(self, address: str, target: object, allowed_methods: list[str]):
        # Supports ipc://, tcp:// (IPv4/IPv6 auto-detected)
        ...

    def start(self):
        ...

    def stop(self):
        ...
```

### 2. DriverClient

```python
# driver_client.py
class DriverClient:
    def __init__(self, request_address: str, control_addresses: list[str]):
        """
        Args:
            request_address: DriverWorkerProxy request address (ipc://)
            control_addresses: Control addresses for all proxies (ipc:// or tcp://)
        """
        self.request_channel = RequestChannelClient(request_address)
        self.control_channel = ControlChannelClient(control_addresses)

    async def generate(self, request: DriverRequest) -> AsyncIterator[DriverRequestOutput]:
        """Async generation with streaming output (auto-starts)"""
        async for output in self.request_channel.submit(request):
            yield output

    def _execute(self, method: str, *args, **kwargs) -> list[Any]:
        """Collective RPC: execute the same method on all workers"""
        return self.control_channel.collective_rpc(method, *args, **kwargs)

    # Convenience methods
    def sleep(self, tag: str = None):
        return self._execute("sleep", tag=tag)

    def wake_up(self, tag: str = None):
        return self._execute("wake_up", tag=tag)

    def close(self):
        """Cleanup resources"""
        self.request_channel.close()
        self.control_channel.close()
```

### 3. DriverWorkerProxy

```python
# driver_worker_proxy.py
class DriverWorkerProxy:
    def __init__(self, config: VExactConfig, request_address: str, control_address: str):
        self.worker = DriverWorker(config)
        self.request_server = RequestChannelServer(
            address=request_address,
            on_request=self._on_request,
            poll_outputs=self._poll_outputs,
        )
        self.control_server = ControlChannelServer(
            address=control_address,
            target=self.worker,
            allowed_methods=["sleep", "wake_up"],
        )

    def start(self):
        self.control_server.start()
        self.request_server.start()
        self.worker.start()

    def _on_request(self, request: DriverRequest):
        """Convert DriverRequest to InferenceRequest and submit"""
        inference_request = InferenceRequest.from_driver_request(request)
        self.worker.submit_request(inference_request)

    def _poll_outputs(self) -> list[DriverRequestOutput]:
        """Poll completed outputs from scheduler"""
        ...
```

### 4. WorkerProxy

```python
# worker_proxy.py
class WorkerProxy:
    def __init__(self, config: VExactConfig, rank: int, control_address: str):
        self.worker = Worker(config, rank)
        self.control_server = ControlChannelServer(
            address=control_address,
            target=self.worker,
            allowed_methods=["sleep", "wake_up"],
        )

    def start(self):
        self.control_server.start()
        self.worker.start()  # blocking
```

## Process Startup Example

```python
# Process 1: DriverWorkerProxy
proxy = DriverWorkerProxy(
    config,
    request_address="ipc:///tmp/vexact_request.sock",
    control_address="ipc:///tmp/vexact_ctrl_0.sock",
)
proxy.start()

# Process 2: WorkerProxy (rank=1)
proxy = WorkerProxy(
    config,
    rank=1,
    control_address="ipc:///tmp/vexact_ctrl_1.sock",
)
proxy.start()

# Process 3: DriverClient
client = DriverClient(
    request_address="ipc:///tmp/vexact_request.sock",
    control_addresses=[
        "ipc:///tmp/vexact_ctrl_0.sock",
        "ipc:///tmp/vexact_ctrl_1.sock",
    ],
)

# Usage (no explicit start needed - auto-starts)
async for output in client.generate(request):
    print(output)

client.sleep(tag="lora_switch")
client.wake_up(tag="lora_switch")

# Cleanup
client.close()
```

## Implementation Details

### Thread-based Design

Both channel servers use **dedicated threads** for simplicity and robustness:

- Thread overhead (\<0.02% of GPU inference time) is negligible
- Simpler than asyncio for this use case
- Independent of main generation loop
- See `thread_vs_asyncio_design.md` for detailed analysis

### Auto-start Pattern

RequestChannelClient auto-starts on first `submit()`:

- No explicit `start()` call needed
- Background receive loop created lazily
- Simpler user API

### Serialization Hooks

DriverRequest provides msgspec hooks for GenerationConfig:

- `DriverRequest.enc_hook()` - converts GenerationConfig to dict
- `DriverRequest.dec_hook()` - converts dict to GenerationConfig
- Used by RequestChannelClient/Server encoders/decoders
