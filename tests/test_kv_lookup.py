import pytest
from refactored_plugins.kv_lookup import ShadowRadixTree

@pytest.mark.asyncio
async def test_shadow_radix_tree():
    """
    Test the manual adding of overlapping blocks and longest_prefix_match functionality
    of the ShadowRadixTree.
    """
    tree = ShadowRadixTree()
    
    # 1. Manually add 3 overlapping blocks
    # Block 1: parent None
    tree.add_block(1, None, [100, 200, 300])
    
    # Block 2: parent is Block 1
    tree.add_block(2, 1, [400, 500, 600])
    
    # Block 3: parent is Block 2
    tree.add_block(3, 2, [700, 800, 900])
    
    # Test 1: Completely new sequence (no match)
    assert tree.longest_prefix_match([999, 888]) == 0
    
    # Test 2: Partially matching sequence
    # Matches tokens from Block 1 and part of Block 2
    assert tree.longest_prefix_match([100, 200, 300, 400, 500, 999]) == 5
    
    # Test 3: Fully matching sequence
    # Matches all tokens across all blocks
    assert tree.longest_prefix_match([100, 200, 300, 400, 500, 600, 700, 800, 900]) == 9
    
    # Test 4: Another partial match targeting only Block 1
    assert tree.longest_prefix_match([100, 200, 300, 999]) == 3
