"""Pure depth-limited expectimax over generic nodes (no chess/torch deps)."""
from __future__ import annotations


def expectimax(node, depth, expand, leaf_value, is_max_node):
    """Return (value, best_move). Max node -> best child's value + its move; chance node ->
    probability-weighted (renormalized) expectation + None; depth<=0 or no children -> leaf."""
    if depth <= 0:
        return leaf_value(node), None
    children = expand(node)  # [(move, child, prob), ...]
    if not children:
        return leaf_value(node), None
    if is_max_node(node):
        best_v, best_m = float("-inf"), None
        for mv, child, _ in children:
            v, _ = expectimax(child, depth - 1, expand, leaf_value, is_max_node)
            if v > best_v:
                best_v, best_m = v, mv
        return best_v, best_m
    total = sum(p for _, _, p in children) or 1.0
    v = 0.0
    for _, child, p in children:
        cv, _ = expectimax(child, depth - 1, expand, leaf_value, is_max_node)
        v += (p / total) * cv
    return v, None
