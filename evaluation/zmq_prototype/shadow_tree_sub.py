#!/usr/bin/env python3
"""
shadow_tree_sub.py

A ZeroMQ Subscriber that connects to tcp://localhost:5557, subscribes to all events,
and maintains a local "shadow_cache" dictionary reflecting SGLang's Radix Tree cache.
When events are received, it updates the shadow cache and prints a beautiful ASCII 
visualization of the live tree hierarchy.

Author: Senior AI Infrastructure Software Engineer
Project: Middleware Token Proxy Middleware - WP2 ZMQ Prototype
"""

import zmq
import json
import sys
from collections import defaultdict

# ANSI Escape Sequences for beautiful terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARK_GRAY = '\033[90m'

def print_radix_tree(shadow_cache):
    """
    Reconstructs and prints the Radix Tree hierarchy from the flat shadow_cache.
    """
    if not shadow_cache:
        print(f"{Colors.DARK_GRAY}  [Cache is currently empty]{Colors.ENDC}")
        return

    # Build adjacency list: parent -> list of child block hashes
    children = defaultdict(list)
    roots = []

    for block_hash, block_data in shadow_cache.items():
        parent = block_data.get("parent_block_hash")
        # A block is a root if its parent is None OR if its parent is not in the shadow cache
        if parent is None or parent not in shadow_cache:
            roots.append(block_hash)
        else:
            children[parent].append(block_hash)

    # Sort roots and children by block hash for deterministic display
    roots.sort()
    for parent in children:
        children[parent].sort()

    total_tokens = sum(len(b.get("token_ids", [])) for b in shadow_cache.values())
    print(f"\n{Colors.BOLD}Shadow Radix Tree Cache State:{Colors.ENDC}")
    print(f"  ├─ {Colors.CYAN}Total Blocks:{Colors.ENDC} {len(shadow_cache)}")
    print(f"  └─ {Colors.CYAN}Total Tokens:{Colors.ENDC} {total_tokens}")
    print(f"{Colors.DARK_GRAY}Tree Visualization:{Colors.ENDC}")

    def dfs(node_hash, prefix="", is_last=True):
        node_data = shadow_cache[node_hash]
        token_ids = node_data.get("token_ids", [])
        
        # Display shortened preview of token IDs
        if len(token_ids) > 6:
            token_preview = f"[{', '.join(map(str, token_ids[:3]))}, ..., {', '.join(map(str, token_ids[-3:]))}]"
        else:
            token_preview = str(token_ids)

        connector = "└── " if is_last else "├── "
        node_label = f"{Colors.GREEN}Block 0x{node_hash:08x}{Colors.ENDC}"
        details = f"{Colors.DARK_GRAY}(tokens: {len(token_ids)}, val: {token_preview}){Colors.ENDC}"

        print(f"{prefix}{connector}{node_label} {details}")

        # Recurse children
        node_children = children[node_hash]
        child_count = len(node_children)
        for i, child_hash in enumerate(node_children):
            new_prefix = prefix + ("    " if is_last else "│   ")
            dfs(child_hash, new_prefix, is_last=(i == child_count - 1))

    # Print the tree starting from each root node
    for idx, root_hash in enumerate(roots):
        dfs(root_hash, prefix="  ", is_last=(idx == len(roots) - 1))
    print()

def main():
    # Setup ZMQ Context and SUB Socket
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    
    connect_address = "tcp://localhost:5557"
    try:
        subscriber.connect(connect_address)
    except Exception as e:
        print(f"{Colors.FAIL}{Colors.BOLD}Failed to connect to {connect_address}: {e}{Colors.ENDC}", file=sys.stderr)
        sys.exit(1)

    # Subscribe to all events (empty prefix string)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    print(f"{Colors.HEADER}{Colors.BOLD}" + "="*60 + f"{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}SGLang Shadow KV Cache Tree Subscriber Started{Colors.ENDC}")
    print(f"{Colors.CYAN}Connecting to:{Colors.ENDC} {connect_address}")
    print(f"{Colors.CYAN}Subscription topic:{Colors.ENDC} [ALL EVENTS]")
    print(f"{Colors.HEADER}{Colors.BOLD}" + "="*60 + f"{Colors.ENDC}")
    print("Waiting for SGLang publisher events... (Press Ctrl+C to terminate)\n")

    # In-memory dictionary tracking cache state: block_hash -> metadata dict
    shadow_cache = {}

    try:
        while True:
            # Receive event string
            event_str = subscriber.recv_string()
            
            try:
                event = json.loads(event_str)
            except json.JSONDecodeError as je:
                print(f"{Colors.FAIL}[ERROR] Failed to parse JSON event: {je}{Colors.ENDC}", file=sys.stderr)
                continue

            event_type = event.get("type")
            seq = event.get("sequence", 0)
            block_hash = event.get("block_hash")

            if not event_type or block_hash is None:
                print(f"{Colors.WARNING}[WARN] Received invalid event structure: {event}{Colors.ENDC}")
                continue

            print(f"{Colors.BLUE}[Seq: {seq:03d}]{Colors.ENDC} Received {Colors.BOLD}{event_type}{Colors.ENDC} for Block {Colors.BOLD}0x{block_hash:08x}{Colors.ENDC}")

            if event_type == "BlockStored":
                # Store the block details in the local shadow tree cache
                shadow_cache[block_hash] = {
                    "parent_block_hash": event.get("parent_block_hash"),
                    "token_ids": event.get("token_ids", []),
                    "block_size": event.get("block_size", 0),
                    "medium": event.get("medium", "GPU"),
                    "seq": seq
                }
            elif event_type == "BlockRemoved":
                # Remove the block from the local shadow tree cache
                if block_hash in shadow_cache:
                    del shadow_cache[block_hash]
                else:
                    print(f"  {Colors.WARNING}* Block 0x{block_hash:08x} was not found in local shadow cache, skipping deletion *{Colors.ENDC}")
            else:
                print(f"  {Colors.WARNING}* Unknown event type: {event_type} *{Colors.ENDC}")

            # Reconstruct and display the current cache tree structure
            print_radix_tree(shadow_cache)
            print("-" * 60)

    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Shutting down subscriber...{Colors.ENDC}")
    finally:
        subscriber.close()
        context.term()
        print(f"{Colors.GREEN}Subscriber terminated cleanly.{Colors.ENDC}")

if __name__ == "__main__":
    main()
