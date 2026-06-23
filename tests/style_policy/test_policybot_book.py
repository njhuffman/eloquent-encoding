import random, chess
from style_policy.opening_book import OpeningBook

class _StubModel:
    def __getattr__(self, _):  # any model attr access -> should NOT happen when book hits
        raise AssertionError("model path used while opening book should have answered")

def test_book_move_short_circuits_model(monkeypatch):
    from style_policy import play
    # Build a bot without loading a real checkpoint: bypass __init__ via __new__
    bot = play.PolicyBot.__new__(play.PolicyBot)
    bot.opening_book = OpeningBook(total_games=100, positions={
        chess.Board().epd(): {"n": 100, "moves": {"e2e4": 100}}})
    bot.book_threshold = 0.01
    bot._book_rng = random.Random(0)
    bot.model = _StubModel()  # explodes if touched
    mv = play.PolicyBot.choose_move(bot, chess.Board())
    assert mv == chess.Move.from_uci("e2e4")

def test_no_book_falls_through(monkeypatch):
    from style_policy import play
    bot = play.PolicyBot.__new__(play.PolicyBot)
    bot.opening_book = None
    bot.book_threshold = 0.01
    bot._book_rng = random.Random(0)
    bot.device = "cpu"
    called = {"hit": False}
    # stub the model path: replace choose_move's model use by monkeypatching the method's tail.
    # Simplest: give a book that returns None and assert lookup path doesn't crash; then
    # verify the bot attempts the model path by raising a sentinel from a stubbed encode.
    class _M:
        def encode(self, *a, **k):
            called["hit"] = True
            raise RuntimeError("model-path-reached")
    bot.model = _M()
    try:
        play.PolicyBot.choose_move(bot, chess.Board())
    except RuntimeError as e:
        assert "model-path-reached" in str(e)
    assert called["hit"]
