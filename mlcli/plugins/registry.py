"""Plugin registry for managing and discovering MLCLI plugins."""

from typing import Dict, List, Optional
from pathlib import Path

from mlcli.plugins.base import PluginBase
from mlcli.core.exceptions import ConfigurationError


class PluginRegistry:
    """Manages plugin discovery, registration, and retrieval.
    
    The registry automatically loads all built-in plugins and provides
    methods to access them by name.
    """
    
    def __init__(self):
        """Initialize registry and load built-in plugins."""
        self._plugins: Dict[str, PluginBase] = {}
        self._load_builtin_plugins()
    
    def _load_builtin_plugins(self):
        """Load all built-in plugins."""
        # Import here to avoid circular dependencies
        try:
            from mlcli.plugins.tabular import TabularPlugin
            self.register(TabularPlugin())
        except ImportError:
            pass  # Plugin not yet implemented
        
        try:
            from mlcli.plugins.chatbot import ChatbotPlugin
            self.register(ChatbotPlugin())
        except ImportError:
            pass  # Plugin not yet implemented
        
        try:
            from mlcli.plugins.image_classification import ImageClassificationPlugin
            self.register(ImageClassificationPlugin())
        except ImportError:
            pass  # Plugin not yet implemented
    
    def register(self, plugin: PluginBase) -> None:
        """Register a plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        self._plugins[plugin.name] = plugin
    
    def get(self, name: str) -> PluginBase:
        """Get plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance
            
        Raises:
            ConfigurationError: If plugin not found
        """
        if name not in self._plugins:
            available = ", ".join(self.list_available())
            raise ConfigurationError(
                f"Plugin '{name}' not found. Available plugins: {available}"
            )
        return self._plugins[name]
    
    def list_available(self) -> List[str]:
        """List all available plugin names.
        
        Returns:
            List of plugin names
        """
        return sorted(list(self._plugins.keys()))
    
    def load_custom_plugins(self, plugin_dir: Path) -> None:
        """Load user-defined plugins from directory.
        
        Args:
            plugin_dir: Directory containing custom plugin modules
            
        Note:
            This is a placeholder for future extensibility.
            Custom plugins must follow the PluginBase interface.
        """
        # TODO: Implement dynamic plugin loading
        pass
