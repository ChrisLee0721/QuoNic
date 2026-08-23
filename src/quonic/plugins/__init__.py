"""Plugin system — extend QuoNic with custom backends, passes, and algorithms.

Example::

    from quonic.plugins import Plugin, register_plugin

    class MyBackend(Plugin):
        name = "my_backend"
        def run(self, circuit, shots=1024, **kwargs):
            ...

    register_plugin(MyBackend())
"""

from .base import AlgorithmPlugin, BackendPlugin, PassPlugin, Plugin
from .registry import get_plugin, list_plugins, register_plugin

__all__ = [
    "AlgorithmPlugin",
    "BackendPlugin",
    "PassPlugin",
    "Plugin",
    "get_plugin",
    "list_plugins",
    "register_plugin",
]
