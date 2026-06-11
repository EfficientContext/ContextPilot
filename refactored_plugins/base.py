from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BasePlugin(ABC):
    """
    Abstract base class for all Token Proxy Plugins.
    Each plugin intercepts a request and performs a specific optimization.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def process(self, request_data: Any) -> Any:
        """
        Process the incoming request data and return the optimized version.
        """
        pass

    @abstractmethod
    def get_plugin_metrics(self) -> Dict[str, float]:
        """
        Return a dictionary of performance and optimization metrics.
        """
        pass


class ContextReorderPlugin(BasePlugin):
    """
    Plugin for reordering prompts to maximize KV Cache prefix sharing.
    Uses ContextPilot clustering and scheduling logic.
    """

    def __init__(self, alpha: float = 0.001, use_gpu: bool = False):
        super().__init__("context_reorder")
        from contextpilot.server.live_index import ContextPilot

        self.pilot = ContextPilot(alpha=alpha, use_gpu=use_gpu)

    async def process(self, request_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Specialized process for batches.
        Note: The Proxy framework will need to handle batching.
        """
        # Extract prompts/messages from batch
        # This is where we'll bridge the OpenAI format to ContextPilot's token lists
        pass
