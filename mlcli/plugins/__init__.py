"""Plugin system for MLCLI.

Enables domain-specific project templates (tabular, chatbot, image classification, etc.)
"""

__all__ = ["PluginBase", "PluginRegistry"]

from mlcli.plugins.base import PluginBase
from mlcli.plugins.registry import PluginRegistry
