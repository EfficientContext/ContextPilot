# Code Review: `mem0-locomo-example` branch (uncommitted changes)

## 🟡 Medium Priority

- [ ] **[BUG-RISK]** `tree_nodes.py:57` — `merge_with()` still does `self.doc_ids = sorted(list(self.content))`, ignoring any prior `ordered_doc_ids`. Not triggered in the current live path, but a latent re-sorting bug if `merge_with` is ever called on a reordered node.
- [ ] **[TESTING]** Add unit test for `ClusterNode(ordered_doc_ids=...)` — verify `node.doc_ids` preserves input order, and verify the default (no `ordered_doc_ids`) still sorts.
- [ ] **[TESTING]** Add unit test for the `_copy_subtree` path — create a node with reordered doc_ids, copy it, and assert the copy's `doc_ids` matches the source order (not sorted).

## 🟢 Low Priority

- [ ] **[STYLE]** `mem0_locomo_example.py:last line` — `import gc; gc.collect()` two statements on one line.
- [ ] **[STYLE]** `mem0_locomo_example.py` — `avg = lambda xs: ...` could be a `def` per PEP 8 E731.
