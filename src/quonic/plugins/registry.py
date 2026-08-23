"""Plugin registry — register and manage plugins.

Example::

    from quonic.plugins import register_plugin, get_plugin, list_plugins

    register_plugin(MyBackend())
    backend = get_plugin("my_backend")
    plugins = list_plugins()
"""

from __future__ import annotations

from .base import Plugin

# Global plugin registry
_PLUGINS: dict[str, Plugin] = {}


def register_plugin(plugin: Plugin) -> None:
    """Register a plugin.

    Args:
        plugin: plugin instance to register

    Raises:
        ValueError: if plugin name is empty or already registered
    """
    if not plugin.name:
        raise ValueError("Plugin must have a name")
    if plugin.name in _PLUGINS:
        raise ValueError(f"Plugin '{plugin.name}' is already registered")
    _PLUGINS[plugin.name] = plugin


def get_plugin(name: str) -> Plugin | None:
    """Get a registered plugin by name.

    Args:
        name: plugin name

    Returns:
        Plugin instance, or None if not found.
    """
    return _PLUGINS.get(name)


def list_plugins() -> list[dict[str, str]]:
    """List all registered plugins.

    Returns:
        List of dicts with name, version, description.
    """
    return [
        {
            "name": p.name,
            "version": p.version,
            "description": p.description,
            "type": type(p).__name__,
        }
        for p in _PLUGINS.values()
    ]


def unregister_plugin(name: str) -> bool:
    """Unregister a plugin by name.

    Args:
        name: plugin name

    Returns:
        True if removed, False if not found.
    """
    if name in _PLUGINS:
        del _PLUGINS[name]
        return True
    return False
