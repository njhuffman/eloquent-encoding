from style_policy.search import expectimax

# toy tree: root (max) -> A (chance), B (chance); A -> Ax,Ay ; B -> Bz
_TREE = {"root": [("a", "A", 0.5), ("b", "B", 0.5)],
         "A": [("x", "Ax", 0.7), ("y", "Ay", 0.3)],
         "B": [("z", "Bz", 1.0)]}
_LEAF = {"Ax": 1.0, "Ay": 0.0, "Bz": 0.5}
_ISMAX = {"root": True, "A": False, "B": False}
_expand = lambda n: _TREE.get(n, [])
_leaf = lambda n: _LEAF.get(n, 0.0)
_ismax = lambda n: _ISMAX.get(n, True)

def test_max_then_chance():
    # depth 2: A=0.7*1+0.3*0=0.7 ; B=1.0*0.5=0.5 ; root max -> 0.7 via "a"
    v, m = expectimax("root", 2, _expand, _leaf, _ismax)
    assert m == "a" and abs(v - 0.7) < 1e-9

def test_depth_limit_stops_at_one():
    # depth 1: children A,B evaluated as leaves (not in _LEAF -> 0.0) -> v 0, move first
    v, m = expectimax("root", 1, _expand, _leaf, _ismax)
    assert v == 0.0 and m == "a"

def test_depth_zero_is_leaf():
    assert expectimax("Ax", 0, _expand, _leaf, _ismax) == (1.0, None)

def test_terminal_empty_expand_is_leaf():
    # Bz has no children -> leaf value even at depth>0
    assert expectimax("Bz", 5, _expand, _leaf, _ismax) == (0.5, None)

def test_chance_renormalizes():
    tree = {"r": [("a", "x", 1.0), ("b", "y", 3.0)]}   # unnormalized weights
    v, _ = expectimax("r", 1, lambda n: tree.get(n, []),
                      lambda n: {"x": 0.0, "y": 1.0}.get(n, 0.0), lambda n: False)
    assert abs(v - 0.75) < 1e-9   # 3/(1+3)
