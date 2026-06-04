import asyncio
import logging
import zmq
import zmq.asyncio
import msgspec
from typing import Any, Dict, List, Optional
from .base import BasePlugin

logger = logging.getLogger(__name__)

class ShadowRadixTree:
    """
    Maintains a shadow copy of the worker's KV cache Radix Tree state
    by mapping block hashes to their parents and token contents.
    """
    def __init__(self):
        # block_hash -> {"parent": parent_hash, "tokens": token_ids}
        self.state: Dict[int, Dict[str, Any]] = {}
        # Cache to speed up prefix matching queries
        self._full_tokens_cache: Dict[int, List[int]] = {}

    def add_block(self, block_hash: int, parent_hash: Optional[int], token_ids: List[int]):
        self.state[block_hash] = {"parent": parent_hash, "tokens": token_ids}
        self._full_tokens_cache.clear()

    def remove_block(self, block_hash: int):
        if block_hash in self.state:
            del self.state[block_hash]
            self._full_tokens_cache.clear()

    def _get_tokens(self, block_hash: int) -> List[int]:
        if block_hash in self._full_tokens_cache:
            return self._full_tokens_cache[block_hash]

        path = []
        curr = block_hash
        while curr is not None and curr in self.state:
            path.append(curr)
            curr = self.state[curr]["parent"]

        tokens = []
        for b_hash in reversed(path):
            tokens.extend(self.state[b_hash]["tokens"])

        self._full_tokens_cache[block_hash] = tokens
        return tokens

    def longest_prefix_match(self, target_token_ids: List[int]) -> int:
        """
        Traverses from root blocks down to find how many tokens match the target.
        Returns the maximum matched tokens.
        """
        best_match = 0
        for block_hash in self.state:
            node_tokens = self._get_tokens(block_hash)
            match_len = 0
            for t1, t2 in zip(target_token_ids, node_tokens):
                if t1 == t2:
                    match_len += 1
                else:
                    break
            if match_len > best_match:
                best_match = match_len
        return best_match


class KVCacheLookupPlugin(BasePlugin):
    """
    Plugin for routing requests to the worker with the highest KV cache prefix match.
    Subscribes to worker ZMQ streams to maintain shadow Radix trees.
    """
    
    def __init__(self, endpoints: List[str], model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        super().__init__("kv_cache_lookup")
        from contextpilot.utils.prompt_generator import get_tokenizer
        
        self.tokenizer = get_tokenizer(model_name)
        if self.tokenizer is None:
            logger.warning(f"Could not load tokenizer for {model_name}. Using fallback char-split.")
            
        self.endpoints = endpoints
        self.trees: Dict[str, ShadowRadixTree] = {endpoint: ShadowRadixTree() for endpoint in endpoints}
        
        self.ctx = zmq.asyncio.Context()
        self.listener_tasks = []
        
        # Spawn ZMQ listener tasks for each endpoint
        for endpoint in endpoints:
            task = asyncio.create_task(self._listen(endpoint))
            self.listener_tasks.append(task)
            
    async def _listen(self, endpoint: str):
        sub = self.ctx.socket(zmq.SUB)
        sub.connect(endpoint)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        
        tree = self.trees[endpoint]
        
        while True:
            try:
                parts = await sub.recv_multipart()
                # SGLang format: topic, seq, msgpack_payload
                if len(parts) >= 3:
                    payload = parts[2]
                elif len(parts) == 1:
                    payload = parts[0]
                else:
                    continue
                    
                event = msgspec.msgpack.decode(payload)
                event_type = event.get("type") or event.get("event_type")
                
                if event_type == "BlockStored":
                    block_hash = event.get("block_hash")
                    parent_hash = event.get("parent_block_hash")
                    token_ids = event.get("token_ids", [])
                    if block_hash is not None:
                        tree.add_block(block_hash, parent_hash, token_ids)
                        
                elif event_type == "BlockRemoved":
                    block_hash = event.get("block_hash")
                    if block_hash is not None:
                        tree.remove_block(block_hash)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ZMQ listener for {endpoint}: {e}")

    def _tokenize(self, text: str) -> List[int]:
        if self.tokenizer:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return [ord(c) for c in text]

    async def process(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize the request messages and query all worker Shadow Radix Trees
        for the longest prefix match. Inject '_route_to' into the request.
        """
        messages = request_data.get("messages", [])
        if not messages:
            return request_data
            
        full_text = "\n".join([m.get("content", "") for m in messages])
        target_tokens = self._tokenize(full_text)
        
        best_endpoint = None
        max_match = -1
        
        for endpoint, tree in self.trees.items():
            match_len = tree.longest_prefix_match(target_tokens)
            if match_len > max_match:
                max_match = match_len
                best_endpoint = endpoint
                
        optimized_request = dict(request_data)
        if best_endpoint:
            optimized_request["_route_to"] = best_endpoint
            
        return optimized_request

    def get_plugin_metrics(self) -> Dict[str, float]:
        return {}
