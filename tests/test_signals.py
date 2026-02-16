"""Tests for threshold.engine.signals — SignalBoard and factories."""

from __future__ import annotations

import pytest

from threshold.engine.signals import (
    Severity,
    Signal,
    SignalBoard,
    SignalType,
    make_amplifier_warning,
    make_bottom_turning,
    make_concentration_warning,
    make_defensive_hold,
    make_eps_rev_sell,
    make_eps_rev_warning,
    make_quant_drop_sell,
    make_quant_freshness_warning,
    make_reversal_confirmed,
    make_sma_breach_sell,
    make_sma_breach_warning,
)

# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

class TestSignal:
    def test_to_legacy_flag(self):
        sig = make_sma_breach_sell(12)
        flag = sig.to_legacy_flag()
        assert flag.startswith("SELL:")
        assert "12 consecutive days" in flag

    def test_serialization_round_trip(self):
        sig = make_quant_drop_sell(-1.5, "2026-01-15")
        d = sig.to_dict()
        restored = Signal.from_dict(d)
        assert restored.signal_type == sig.signal_type
        assert restored.severity == sig.severity
        assert restored.message == sig.message
        assert restored.legacy_prefix == sig.legacy_prefix
        assert restored.metadata == sig.metadata

    def test_frozen(self):
        sig = make_sma_breach_sell(10)
        with pytest.raises(AttributeError):
            sig.message = "mutated"  # type: ignore


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

class TestFactories:
    def test_sma_breach_sell(self):
        sig = make_sma_breach_sell(15)
        assert sig.signal_type == SignalType.SELL_HARD
        assert sig.severity == Severity.HIGH
        assert sig.metadata["days_below"] == 15

    def test_sma_breach_warning(self):
        sig = make_sma_breach_warning(8)
        assert sig.signal_type == SignalType.EARLY_WARNING
        assert sig.metadata["days_below"] == 8

    def test_quant_drop_sell(self):
        sig = make_quant_drop_sell(-1.2, "2026-01-10")
        assert sig.signal_type == SignalType.SELL_HARD
        assert sig.metadata["delta"] == -1.2
        assert sig.metadata["compare_date"] == "2026-01-10"

    def test_eps_rev_sell(self):
        sig = make_eps_rev_sell(3.5, -0.269)
        assert sig.signal_type == SignalType.SELL_HARD
        assert "sub-grades" in sig.message

    def test_eps_rev_warning(self):
        sig = make_eps_rev_warning(2.0, -0.154)
        assert sig.signal_type == SignalType.EARLY_WARNING
        assert "trigger at 3" in sig.message

    def test_quant_freshness_warning(self):
        sig = make_quant_freshness_warning()
        assert sig.signal_type == SignalType.VERIFY
        assert sig.severity == Severity.INFO

    def test_defensive_hold(self):
        sig = make_defensive_hold("HEDGE", -0.85)
        assert sig.signal_type == SignalType.HOLD_OVERRIDE
        assert sig.metadata["classification"] == "HEDGE"

    def test_amplifier_warning(self):
        sig = make_amplifier_warning(1.78)
        assert sig.signal_type == SignalType.TRIM_PRIORITY
        assert sig.metadata["downside_capture"] == 1.78

    def test_reversal_confirmed(self):
        sig = make_reversal_confirmed()
        assert sig.signal_type == SignalType.BUY_CONFIRMED

    def test_bottom_turning(self):
        sig = make_bottom_turning()
        assert sig.signal_type == SignalType.BUY_WATCHLIST

    def test_concentration_warning(self):
        sig = make_concentration_warning(["AAPL", "MSFT", "GOOG"], 12.0)
        assert sig.signal_type == SignalType.DEPLOYMENT_GATE
        assert "AAPL" in sig.message
        assert sig.metadata["effective_bets"] == 12.0


# ---------------------------------------------------------------------------
# SignalBoard
# ---------------------------------------------------------------------------

class TestSignalBoard:
    def test_empty_board(self):
        board = SignalBoard()
        assert len(board) == 0
        assert not board
        assert board.net_action == "NONE"

    def test_add_and_len(self):
        board = SignalBoard()
        board.add(make_sma_breach_sell(12))
        assert len(board) == 1
        assert bool(board) is True

    def test_sells_filter(self):
        board = SignalBoard()
        board.add(make_sma_breach_sell(12))
        board.add(make_sma_breach_warning(8))
        board.add(make_quant_drop_sell(-1.5, "2026-01-10"))
        assert len(board.sells) == 2
        assert len(board.warnings) == 1

    def test_buy_signals_filter(self):
        board = SignalBoard()
        board.add(make_reversal_confirmed())
        board.add(make_bottom_turning())
        assert len(board.buy_signals) == 2

    def test_to_legacy_flags(self):
        board = SignalBoard()
        board.add(make_sma_breach_sell(12))
        board.add(make_quant_drop_sell(-1.5, "2026-01-10"))
        flags = board.to_legacy_flags()
        assert len(flags) == 2
        assert all(isinstance(f, str) for f in flags)
        assert flags[0].startswith("SELL:")
        assert flags[1].startswith("SELL:")

    def test_serialization_round_trip(self):
        board = SignalBoard()
        board.add(make_sma_breach_sell(12))
        board.add(make_reversal_confirmed())
        data = board.to_dict()
        restored = SignalBoard.from_dict(data)
        assert len(restored) == 2
        assert restored.sells[0].metadata["days_below"] == 12

    def test_repr(self):
        board = SignalBoard()
        r = repr(board)
        assert "SignalBoard" in r
        assert "0 signals" in r


# ---------------------------------------------------------------------------
# Net Action Priority Resolution
# ---------------------------------------------------------------------------

class TestNetAction:
    def test_two_sells_review(self):
        board = SignalBoard()
        board.add(make_sma_breach_sell(12))
        board.add(make_quant_drop_sell(-1.5, "2026-01-10"))
        assert board.net_action == "REVIEW"

    def test_sell_plus_hold_override(self):
        board = SignalBoard()
        board.add(make_sma_breach_sell(12))
        board.add(make_defensive_hold("HEDGE", -0.85))
        assert board.net_action == "HOLD"

    def test_single_sell_watch(self):
        board = SignalBoard()
        board.add(make_sma_breach_sell(12))
        assert board.net_action == "WATCH"

    def test_trim_no_sells(self):
        board = SignalBoard()
        board.add(make_amplifier_warning(1.78))
        assert board.net_action == "TRIM"

    def test_buy_confirmed(self):
        board = SignalBoard()
        board.add(make_reversal_confirmed())
        assert board.net_action == "BUY"

    def test_buy_watchlist(self):
        board = SignalBoard()
        board.add(make_bottom_turning())
        assert board.net_action == "WATCHLIST"

    def test_warnings_only_watch(self):
        board = SignalBoard()
        board.add(make_sma_breach_warning(8))
        assert board.net_action == "WATCH"

    def test_has_sell_review_two(self):
        board = SignalBoard()
        board.add(make_sma_breach_sell(12))
        board.add(make_eps_rev_sell(3.5, -0.269))
        assert board.has_sell_review is True

    def test_has_sell_review_one(self):
        board = SignalBoard()
        board.add(make_sma_breach_sell(12))
        assert board.has_sell_review is False
