from typing import List
import numpy as np

def generate_contexts(num_contexts: int, 
                     avg_chunks: int = 20,
                     chunk_variance: int = 5,
                     chunk_pool_size: int = 10000,
                     seed: int = 42) -> List[List[int]]:
    """Generate contexts with specified average length."""
    np.random.seed(seed)
    contexts = []
    
    print(f"Generating {num_contexts:,} contexts (avg length={avg_chunks})...")
    for i in range(num_contexts):
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1:,}/{num_contexts:,} contexts...")
        
        num_chunks = max(5, int(np.random.normal(avg_chunks, chunk_variance)))
        chunk_ids = np.random.zipf(1.5, num_chunks) % chunk_pool_size
        chunk_ids = list(set(chunk_ids))
        contexts.append(chunk_ids)
    
    print(f"✓ Generated {num_contexts:,} contexts")
    print(f"  Avg chunks per context: {np.mean([len(c) for c in contexts]):.1f}")
    return contexts