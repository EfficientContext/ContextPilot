"""
ContextPilot Index HTTP Client

Simple client for calling the ContextPilot Index Server from SGLang.
This is what SGLang should use to sync eviction with the remote index.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    import requests
except ImportError:
    raise ImportError(
        "requests library is required for the HTTP client. "
        "Install with: pip install requests"
    )

logger = logging.getLogger(__name__)


class ContextPilotIndexClient:
    """
    Client for ContextPilot Live Index Server.
    
    This is what SGLang should instantiate to communicate with the index server.
    
    Example usage in SGLang:
        # In scheduler initialization:
        self.contextpilot_client = ContextPilotIndexClient("http://localhost:8765")

        # In eviction callback:
        def on_cache_evict(self, evicted_request_ids):
            # Sync eviction with ContextPilot index
            self.contextpilot_client.evict(evicted_request_ids)
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8765",
        timeout: float = 1.0,
        retry_on_failure: bool = False
    ):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the ContextPilot index server
            timeout: Request timeout in seconds (default 1.0 for low latency)
            retry_on_failure: Whether to retry on network failures
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retry_on_failure = retry_on_failure
        self.session = requests.Session()
    
    def _post(self, endpoint: str, json_data: Dict) -> Optional[Dict]:
        """Make a POST request to the server."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.post(
                url,
                json=json_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            logger.warning(f"ContextPilot index request timed out: {endpoint}")
            return None
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"ContextPilot index request failed: {e}")
            return None
    
    def _get(self, endpoint: str) -> Optional[Dict]:
        """Make a GET request to the server."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            logger.warning(f"ContextPilot index request timed out: {endpoint}")
            return None
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"ContextPilot index request failed: {e}")
            return None
    
    def evict(self, request_ids: List[str]) -> Optional[Dict[str, Any]]:
        """
        Evict requests from the index.

        THIS IS THE MAIN METHOD THAT SGLANG SHOULD CALL FOR EVICTION SYNC.

        Args:
            request_ids: List of request IDs to evict (from SGLang's cache eviction)

        Returns:
            Dictionary with eviction results:
            - removed_count: Number of requests successfully removed
            - not_found: List of request IDs that were not in the index
            - conversations_cleared: Number of conversation chains cleared
            Returns None if request failed
        """
        return self._post("/evict", {"request_ids": request_ids})
    
    def search(
        self, 
        context: List[int], 
        update_access: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Search for a context in the index."""
        return self._post("/search", {
            "context": context,
            "update_access": update_access
        })
    
    def update_node(
        self, 
        search_path: List[int], 
        token_delta: int
    ) -> Optional[Dict[str, Any]]:
        """Update a node's token count."""
        return self._post("/update", {
            "search_path": search_path,
            "token_delta": token_delta
        })
    
    def insert(
        self, 
        context: List[int], 
        search_path: List[int], 
        total_tokens: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Insert a new context.
        
        Returns a dictionary containing:
        - node_id: The new leaf node ID
        - search_path: Path to the new node
        - request_id: Auto-generated request_id for token tracking
        """
        return self._post("/insert", {
            "context": context,
            "search_path": search_path,
            "total_tokens": total_tokens
        })
    
    # =========================================================================
    # Primary API — works in both stateless and stateful server modes
    # =========================================================================

    def reorder(
        self,
        contexts: List[List],
        alpha: float = 0.001,
        use_gpu: bool = False,
        linkage_method: str = "average",
        initial_tokens_per_context: int = 0,
        deduplicate: bool = False,
        parent_request_ids: Optional[List[Optional[str]]] = None,
        hint_template: Optional[str] = None,
    ):
        """
        Reorder contexts for optimal prefix sharing.

        Calls the unified ``POST /reorder`` endpoint, which auto-dispatches
        based on server mode (stateless vs stateful).

        Args:
            contexts: List of contexts (each is a list of document IDs or strings)
            alpha: Distance computation parameter (default 0.001)
            use_gpu: Use GPU for distance computation (default False)
            linkage_method: Linkage method for clustering (default "average")
            initial_tokens_per_context: Initial token count per context (stateful only)
            deduplicate: Enable multi-turn deduplication (stateful only)
            parent_request_ids: Parent request IDs for deduplication (stateful only)
            hint_template: Custom template for reference hints (stateful only)

        Returns:
            A tuple ``(reordered_contexts, execution_order)`` where

            * **reordered_contexts** – contexts reordered for maximum prefix sharing.
            * **execution_order** – list of original context indices so that
              ``reordered_contexts[i]`` corresponds to ``contexts[execution_order[i]]``.

            Returns ``None`` if the request failed.
        """
        payload: Dict[str, Any] = {
            "contexts": contexts,
            "alpha": alpha,
            "use_gpu": use_gpu,
            "linkage_method": linkage_method,
            "initial_tokens_per_context": initial_tokens_per_context,
            "deduplicate": deduplicate,
        }
        if parent_request_ids is not None:
            payload["parent_request_ids"] = parent_request_ids
        if hint_template is not None:
            payload["hint_template"] = hint_template

        result = self._post("/reorder", payload)
        if result is None:
            return None
        return result["reordered_contexts"], result["original_indices"]

    def reorder_raw(
        self,
        contexts: List[List],
        alpha: float = 0.001,
        use_gpu: bool = False,
        linkage_method: str = "average",
        initial_tokens_per_context: int = 0,
        deduplicate: bool = False,
        parent_request_ids: Optional[List[Optional[str]]] = None,
        hint_template: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Same as :py:meth:`reorder` but returns the full server response dict
        (includes ``request_ids``, ``groups``, ``stats``, ``deduplication``, etc.).
        """
        payload: Dict[str, Any] = {
            "contexts": contexts,
            "alpha": alpha,
            "use_gpu": use_gpu,
            "linkage_method": linkage_method,
            "initial_tokens_per_context": initial_tokens_per_context,
            "deduplicate": deduplicate,
        }
        if parent_request_ids is not None:
            payload["parent_request_ids"] = parent_request_ids
        if hint_template is not None:
            payload["hint_template"] = hint_template

        return self._post("/reorder", payload)

    # =========================================================================
    # Legacy wrappers (deprecated — use reorder / reorder_raw)
    # =========================================================================

    def build(
        self,
        contexts: List[List[int]],
        alpha: float = 0.001,
        use_gpu: bool = False,
        linkage_method: str = "average",
        initial_tokens_per_context: int = 0,
        deduplicate: bool = False,
        parent_request_ids: Optional[List[Optional[str]]] = None,
        hint_template: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Deprecated — use :py:meth:`reorder` or :py:meth:`reorder_raw` instead."""
        return self.reorder_raw(
            contexts=contexts,
            alpha=alpha,
            use_gpu=use_gpu,
            linkage_method=linkage_method,
            initial_tokens_per_context=initial_tokens_per_context,
            deduplicate=deduplicate,
            parent_request_ids=parent_request_ids,
            hint_template=hint_template,
        )

    def schedule(
        self,
        contexts: List[List[int]],
        alpha: float = 0.001,
        use_gpu: bool = False,
        linkage_method: str = "average"
    ) -> Optional[Dict[str, Any]]:
        """Deprecated — use :py:meth:`reorder` or :py:meth:`reorder_raw` instead."""
        return self.reorder_raw(
            contexts=contexts,
            alpha=alpha,
            use_gpu=use_gpu,
            linkage_method=linkage_method,
        )

    # =========================================================================
    # Multi-Turn Deduplication
    # =========================================================================

    def deduplicate(
        self,
        contexts: List[List[int]],
        parent_request_ids: List[Optional[str]],
        hint_template: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Deduplicate contexts for multi-turn conversations.

        Lightweight endpoint for Turn 2+ — no index build or search.

        Recommended flow:
        1. Turn 1: Call ``reorder(deduplicate=True)``
        2. Turn 2+: Call ``deduplicate()`` (10-100x faster)

        Args:
            contexts: List of contexts to deduplicate
            parent_request_ids: Parent request IDs (None = new conversation)
            hint_template: Custom template for reference hints

        Returns:
            Dictionary with request_ids, results, summary
        """
        payload: Dict[str, Any] = {
            "contexts": contexts,
            "parent_request_ids": parent_request_ids,
        }
        if hint_template is not None:
            payload["hint_template"] = hint_template

        return self._post("/deduplicate", payload)

    # =========================================================================
    # Utility
    # =========================================================================

    def reset(self) -> Optional[Dict[str, Any]]:
        """Reset the index and conversation tracker."""
        return self._post("/reset", {})

    def get_requests(self) -> Optional[Dict[str, Any]]:
        """Get all tracked request IDs."""
        return self._get("/requests")

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get index statistics."""
        return self._get("/stats")

    def health(self) -> Optional[Dict[str, Any]]:
        """Check server health."""
        return self._get("/health")

    def is_ready(self) -> bool:
        """Check if the server is ready."""
        health = self.health()
        return health is not None and health.get("status") == "ready"
    
    def close(self):
        """Close the client session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.close()


# Convenience functions for simple usage

def evict_requests(
    request_ids: List[str],
    server_url: str = "http://localhost:8765"
) -> Optional[Dict[str, Any]]:
    """
    Simple function to evict requests from the index.

    For one-off calls without maintaining a client instance.

    Args:
        request_ids: List of request IDs to evict
        server_url: ContextPilot server URL

    Returns:
        Dictionary with removed_count, not_found, conversations_cleared
    """
    try:
        response = requests.post(
            f"{server_url}/evict",
            json={"request_ids": request_ids},
            timeout=1.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"ContextPilot eviction failed: {e}")
        return None





def schedule_batch(
    contexts: List[List[int]],
    server_url: str = "http://localhost:8765",
    alpha: float = 0.001,
    use_gpu: bool = False,
    linkage_method: str = "average",
    timeout: float = 30.0
):
    """
    Reorder a batch of contexts for optimal prefix sharing (convenience function).
    
    For one-off calls without maintaining a client instance.
    Works with both stateless and stateful server modes.
    
    Args:
        contexts: List of contexts (each is a list of document IDs)
        server_url: ContextPilot server URL
        alpha: Distance computation parameter
        use_gpu: Use GPU for distance computation
        linkage_method: Linkage method for clustering
        timeout: Request timeout (longer for large batches)
    
    Returns:
        Dictionary with reordered_contexts, original_indices, groups
    """
    try:
        response = requests.post(
            f"{server_url}/reorder",
            json={
                "contexts": contexts,
                "alpha": alpha,
                "use_gpu": use_gpu,
                "linkage_method": linkage_method
            },
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"ContextPilot reorder failed: {e}")
        return None
