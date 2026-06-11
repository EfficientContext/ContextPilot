import logging
import time
from typing import Any, Dict
from .base import BasePlugin

logger = logging.getLogger(__name__)


class SkillAwareContextPlugin(BasePlugin):
    """
    Plugin for dynamically injecting OpenAI tool schemas based on the required
    skills specified by the framework's router.
    """

    def __init__(self, tool_registry: Dict[str, Dict]):
        super().__init__("skill_aware_context")
        self.tool_registry = tool_registry

        # Telemetry variables
        self.total_tools_filtered = 0
        self.last_execution_time_ms = 0.0

    async def process(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the incoming request and inject the required tool schemas.
        """
        start_time = time.perf_counter()
        optimized_request = dict(request_data)

        required_skills = optimized_request.get("_required_skills")
        if required_skills is not None:
            injected_tools = []
            for skill in required_skills:
                if skill in self.tool_registry:
                    injected_tools.append(self.tool_registry[skill])
                else:
                    logger.warning(f"Skill '{skill}' requested but not found in tool registry.")

            # OpenAI requires a 'tools' array
            optimized_request["tools"] = injected_tools

            # Update telemetry
            # Number of tools filtered out = total available - total injected
            filtered_count = len(self.tool_registry) - len(injected_tools)
            self.total_tools_filtered += filtered_count

        self.last_execution_time_ms = (time.perf_counter() - start_time) * 1000
        return optimized_request

    def get_plugin_metrics(self) -> Dict[str, float]:
        """
        Return the telemetry data.
        """
        return {
            "total_tools_filtered": float(self.total_tools_filtered),
            "last_execution_time_ms": self.last_execution_time_ms,
        }
