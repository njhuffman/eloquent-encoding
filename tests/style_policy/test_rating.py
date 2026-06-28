from style_policy.rating import expected_score, implied_rating, score_ci, mle_rating

def test_expected_score():
    assert abs(expected_score(1500, 1500) - 0.5) < 1e-9
    assert expected_score(1500, 1700) > 0.5   # stronger than the anchor
    assert expected_score(1900, 1500) < 0.5

def test_implied_rating():
    assert abs(implied_rating(0.5, 1500) - 1500) < 1e-6
    assert implied_rating(0.75, 1500) > 1500
    assert implied_rating(0.25, 1500) < 1500

def test_mle_recovers_known_rating():
    true_b = 1600
    rows = [(a, 1000, expected_score(a, true_b)) for a in (1100, 1300, 1500, 1700, 1900)]
    r, se = mle_rating(rows)
    assert abs(r - true_b) < 5
    assert se < 30

def test_mle_se_shrinks_with_n():
    small = [(a, 50, expected_score(a, 1600)) for a in (1300, 1500, 1700)]
    big = [(a, 5000, expected_score(a, 1600)) for a in (1300, 1500, 1700)]
    assert mle_rating(big)[1] < mle_rating(small)[1]

def test_score_ci_bounds():
    s, lo, hi = score_ci(50, 0, 50)
    assert abs(s - 0.5) < 1e-9 and 0.0 <= lo <= s <= hi <= 1.0
