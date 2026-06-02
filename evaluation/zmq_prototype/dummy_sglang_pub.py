#!/usr/bin/env python3
"""
dummy_sglang_pub.py

A ZeroMQ Publisher that simulates SGLang's internal Radix Tree KV Cache event stream.
It binds to tcp://*:5557 and randomly publishes BlockStored and BlockRemoved events
in a logically consistent manner (maintaining a simulated tree structure) every second.

Author: Senior AI Infrastructure Software Engineer
Project: Middleware Token Proxy Middleware - WP2 ZMQ Prototype
"""

import zmq
import json
import time
import random
import sys

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

def log_info(msg):
    print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {msg}")

def log_success(msg):
    print(f"{Colors.GREEN}[STORED]{Colors.ENDC} {msg}")

def log_warning(msg):
    print(f"{Colors.WARNING}[REMOVED]{Colors.ENDC} {msg}")

def main():
    # Setup ZMQ Context and PUB Socket
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    
    # Configure high-water mark to prevent memory bloat
    publisher.set_hwm(1000)
    
    bind_address = "tcp://*:5557"
    try:
        publisher.bind(bind_address)
    except Exception as e:
        print(f"{Colors.FAIL}{Colors.BOLD}Failed to bind to {bind_address}: {e}{Colors.ENDC}", file=sys.stderr)
        sys.exit(1)

    print(f"{Colors.HEADER}{Colors.BOLD}" + "="*60 + f"{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}SGLang Dummy KV Cache Event Publisher Started{Colors.ENDC}")
    print(f"{Colors.CYAN}Binding Address:{Colors.ENDC} {bind_address}")
    print(f"{Colors.CYAN}Simulation Rate:{Colors.ENDC} 1 event / sec")
    print(f"{Colors.HEADER}{Colors.BOLD}" + "="*60 + f"{Colors.ENDC}")
    print("Press Ctrl+C to terminate the publisher gracefully.\n")

    # Local state to ensure logically consistent events
    # Maps block_hash -> parent_block_hash
    active_blocks = {}
    
    # Track sequence number to simulate SGLang's monotonic event sequence
    sequence_number = 0

    try:
        while True:
            # We want to maintain a reasonable number of blocks in the cache (e.g., 3 to 15)
            num_blocks = len(active_blocks)
            
            # Decide whether to store a new block or remove an existing one
            # If cache is empty, we must store. If cache is full (> 15), we prefer to remove.
            if num_blocks == 0:
                action = "store"
            elif num_blocks > 12:
                action = "remove" if random.random() < 0.7 else "store"
            else:
                # 65% chance to store (grow), 35% chance to remove (evict)
                action = "store" if random.random() < 0.65 else "remove"

            sequence_number += 1

            if action == "store":
                # Generate a unique block hash (simulating a 64-bit integer hash)
                block_hash = random.randint(10000000, 99999999)
                while block_hash in active_blocks:
                    block_hash = random.randint(10000000, 99999999)

                # Determine parent block hash (simulate tree branching)
                parent_block_hash = None
                if active_blocks and random.random() < 0.7:
                    # Pick an existing block as parent to create a hierarchy
                    parent_block_hash = random.choice(list(active_blocks.keys()))

                # Generate a list of random token IDs (representing the prefix contents of this block)
                # SGLang usually has a block size / page size (e.g. 16 tokens)
                block_size = 16
                token_ids = [random.randint(1, 50000) for _ in range(block_size)]

                # Construct BlockStored event payload
                event = {
                    "type": "BlockStored",
                    "sequence": sequence_number,
                    "block_hash": block_hash,
                    "parent_block_hash": parent_block_hash,
                    "token_ids": token_ids,
                    "block_size": block_size,
                    "medium": "GPU"
                }

                # Update local tracking
                active_blocks[block_hash] = parent_block_hash

                # Publish event
                event_str = json.dumps(event)
                publisher.send_string(event_str)

                parent_str = f"0x{parent_block_hash:08x}" if parent_block_hash else "None"
                log_success(f"Block: 0x{block_hash:08x} | Parent: {parent_str} | Tokens: {token_ids[:3]}... ({block_size} tokens)")

            else:  # remove
                # Pick a block to remove
                # To simulate realistic tree eviction, we should ideally evict leaf blocks.
                # Let's find leaf blocks (blocks that are not parents of any other active blocks)
                all_parents = set(active_blocks.values())
                leaves = [b for b in active_blocks if b not in all_parents]

                # Fallback to any active block if no leaves are easily found (should always find at least one)
                block_to_remove = random.choice(leaves) if leaves else random.choice(list(active_blocks.keys()))

                # Construct BlockRemoved event payload
                event = {
                    "type": "BlockRemoved",
                    "sequence": sequence_number,
                    "block_hash": block_to_remove,
                    "medium": "GPU"
                }

                # Update local tracking
                del active_blocks[block_to_remove]

                # Publish event
                event_str = json.dumps(event)
                publisher.send_string(event_str)

                log_warning(f"Block: 0x{block_to_remove:08x}")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Shutting down publisher...{Colors.ENDC}")
    finally:
        publisher.close()
        context.term()
        print(f"{Colors.GREEN}Publisher terminated cleanly.{Colors.ENDC}")

if __name__ == "__main__":
    main()
