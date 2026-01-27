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
        
        # In eviction code:
        def evict_tokens(self, num_tokens):
            self.tree_cache.evict(num_tokens)
            self.contextpilot_client.evict(num_tokens)  # Sync with index
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
    
    def evict(self, num_tokens: int) -> Optional[Dict[str, Any]]:
        """
        Evict tokens from the index.
        
        THIS IS THE MAIN METHOD THAT SGLANG SHOULD CALL FOR EVICTION SYNC.
        
        Args:
            num_tokens: Number of tokens to evict (same as SGLang's eviction)
        
        Returns:
            Dictionary with eviction results, or None if request failed
        """
        return self._post("/evict", {"num_tokens": num_tokens})
    
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
    # Index Building (Stateful Mode)
    # =========================================================================
    
    def build(
        self,
        contexts: List[List[int]],
        alpha: float = 0.005,
        use_gpu: bool = False,
        linkage_method: str = "average",
        initial_tokens_per_context: int = 0,
        incremental: bool = False,
        deduplicate: bool = False,
        parent_request_ids: Optional[List[Optional[str]]] = None,
        hint_template: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Build a new index or incrementally update an existing one.
        
        This is the main method for creating the live index.
        
        Args:
            contexts: List of contexts (each is a list of document IDs)
            alpha: Distance computation parameter (default 0.005)
            use_gpu: Use GPU for distance computation (default False)
            linkage_method: Clustering method (default "average")
            initial_tokens_per_context: Initial token count per context
            incremental: Use incremental build (search/reorder/merge)
            deduplicate: Enable multi-turn deduplication
            parent_request_ids: Parent request IDs for deduplication
            hint_template: Custom template for reference hints
            
        Returns:
            Dictionary with request_ids, stats, and optional deduplication results
        """
        payload = {
            "contexts": contexts,
            "alpha": alpha,
            "use_gpu": use_gpu,
            "linkage_method": linkage_method,
            "initial_tokens_per_context": initial_tokens_per_context,
            "incremental": incremental,
            "deduplicate": deduplicate,
        }
        
        if parent_request_ids is not None:
            payload["parent_request_ids"] = parent_request_ids
        if hint_template is not None:
            payload["hint_template"] = hint_template
            
        return self._post("/build", payload)
    
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
        
        This is a lightweight endpoint for Turn 2+ in conversations.
        It only performs deduplication - no index build or search operations.
        
        Recommended flow:
        1. Turn 1: Call build() with deduplicate=True
        2. Turn 2+: Call deduplicate() (10-100x faster than build)
        
        Args:
            contexts: List of contexts to deduplicate
            parent_request_ids: Parent request IDs (None = new conversation)
            hint_template: Custom template for reference hints
            
        Returns:
            Dictionary with:
            - request_ids: New request IDs for this turn
            - results: List of deduplication results per context
            - summary: Statistics about deduplication
        """
        payload = {
            "contexts": contexts,
            "parent_request_ids": parent_request_ids,
        }
        
        if hint_template is not None:
            payload["hint_template"] = hint_template
            
        return self._post("/deduplicate", payload)
    
    def reset(self) -> Optional[Dict[str, Any]]:
        """
        Reset the index and conversation tracker.
        
        Call this to clear all state and start fresh.
        
        Returns:
            Dictionary with reset confirmation
        """
        return self._post("/reset", {})
    
    # =========================================================================
    # Token Tracking (SGLang Integration)
    # =========================================================================
    
    def update_tokens(
        self,
        request_id: str,
        num_tokens: int
    ) -> Optional[Dict[str, Any]]:
        """
        Update token count for a request.
        
        THIS IS THE MAIN METHOD FOR SGLANG INTEGRATION.
        
        Call this when:
        1. A request starts processing (with initial input tokens)
        2. When generation completes (with total tokens: input + output)
        
        The method:
        - Updates the token count for the given request_id
        - Automatically triggers eviction if capacity is exceeded
        - Returns eviction results if any nodes were evicted
        
        Args:
            request_id: The request ID (from build or insert response)
            num_tokens: Total number of tokens for this request
            
        Returns:
            Dictionary with update result and any eviction info
        """
        return self._post("/update_tokens", {
            "request_id": request_id,
            "num_tokens": num_tokens
        })
    
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
    
    # =========================================================================
    # Stateless Mode (Batch Scheduling Only)
    # =========================================================================
    
    def schedule(
        self,
        contexts: List[List[int]],
        alpha: float = 0.005,
        use_gpu: bool = False,
        linkage_method: str = "average"
    ) -> Optional[Dict[str, Any]]:
        """
        Schedule a batch of contexts (STATELESS MODE).
        
        This performs clustering and scheduling WITHOUT tracking cache state.
        Use this when you want to:
        1. Just reorder contexts for optimal prefix sharing
        2. Process batches independently without maintaining server state
        3. Send scheduled contexts directly to LLM engine without cache sync
        
        No cache tracking, eviction, or token updates required.
        Each call is independent - perfect for batch processing.
        
        Args:
            contexts: List of contexts (each is a list of document IDs)
            alpha: Distance computation parameter (default 0.005)
            use_gpu: Use GPU for distance computation (default False)
            linkage_method: Linkage method for clustering (default "average")
        
        Returns:
            Dictionary with:
            - scheduled_contexts: Reordered contexts for optimal prefix sharing
            - original_indices: Mapping to original context indices
            - groups: Execution groups with prefix sharing info
        """
        return self._post("/schedule", {
            "contexts": contexts,
            "alpha": alpha,
            "use_gpu": use_gpu,
            "linkage_method": linkage_method
        })
    
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

def evict_tokens(num_tokens: int, server_url: str = "http://localhost:8765"):
    """
    Simple function to evict tokens.
    
    For one-off calls without maintaining a client instance.
    """
    try:
        response = requests.post(
            f"{server_url}/evict",
            json={"num_tokens": num_tokens},
            timeout=1.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"ContextPilot eviction failed: {e}")
        return None


def update_request_tokens(
    request_id: str,
    input_tokens: int,
    output_tokens: int,
    server_url: str = "http://localhost:8765"
):
    """
    Simple function to update request tokens.
    
    For one-off calls without maintaining a client instance.
    """
    try:
        response = requests.post(
            f"{server_url}/update_request_tokens",
            json={
                "request_id": request_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            },
            timeout=1.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"ContextPilot token update failed: {e}")
        return None


def schedule_batch(
    contexts: List[List[int]],
    server_url: str = "http://localhost:8765",
    alpha: float = 0.005,
    use_gpu: bool = False,
    linkage_method: str = "average",
    timeout: float = 30.0
):
    """
    Schedule a batch of contexts for optimal prefix sharing (STATELESS MODE).
    
    For one-off calls without maintaining a client instance.
    Perfect for batch processing without needing to track cache state.
    
    Args:
        contexts: List of contexts (each is a list of document IDs)
        server_url: ContextPilot server URL
        alpha: Distance computation parameter
        use_gpu: Use GPU for distance computation
        linkage_method: Linkage method for clustering
        timeout: Request timeout (longer for large batches)
    
    Returns:
        Dictionary with scheduled_contexts, original_indices, groups
    """
    try:
        response = requests.post(
            f"{server_url}/schedule",
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
        logger.warning(f"ContextPilot batch scheduling failed: {e}")
        return None
