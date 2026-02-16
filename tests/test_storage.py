"""Tests for the SQLite storage layer."""

from __future__ import annotations

from threshold.storage.database import Database
from threshold.storage.migrations import ensure_schema
from threshold.storage.queries import (
    delete_ticker,
    get_data_freshness,
    get_drawdown_classifications,
    get_latest_scores,
    get_latest_scoring_run,
    get_score_history,
    get_ticker,
    get_ticker_count,
    insert_score,
    insert_scoring_run,
    insert_signal,
    list_accounts,
    list_scoring_runs,
    list_tickers,
    update_data_freshness,
    update_scoring_run,
    upsert_account,
    upsert_drawdown_classification,
    upsert_ticker,
)


class TestDatabase:
    """Test Database connection and basic operations."""

    def test_connect_creates_file(self, tmp_path):
        db_path = tmp_path / "test.db"
        assert not db_path.exists()
        db = Database(db_path)
        db.connect()
        assert db_path.exists()
        db.close()

    def test_context_manager(self, tmp_path):
        with Database(tmp_path / "test.db") as db:
            db.execute("SELECT 1")

    def test_schema_version_empty(self, tmp_path):
        db = Database(tmp_path / "test.db")
        assert db.schema_version() == 0
        db.close()


class TestMigrations:
    """Test migration system."""

    def test_initial_migration(self, test_db):
        assert test_db.schema_version() >= 1

    def test_tables_exist(self, test_db):
        tables = test_db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {r["name"] for r in tables}
        expected = {
            "_schema_version", "tickers", "accounts", "positions",
            "scoring_runs", "scores", "signals", "drawdown_classifications",
            "watchlists", "data_freshness", "grace_periods",
            "alden_categories", "alerts",
        }
        assert expected.issubset(table_names)

    def test_idempotent(self, test_db):
        v1 = ensure_schema(test_db)
        v2 = ensure_schema(test_db)
        assert v1 == v2


class TestTickerQueries:
    """Test ticker CRUD operations."""

    def test_upsert_and_get(self, test_db):
        upsert_ticker(test_db, "AAPL", name="Apple Inc.", type="stock", sector="Technology")
        t = get_ticker(test_db, "AAPL")
        assert t is not None
        assert t["name"] == "Apple Inc."
        assert t["sector"] == "Technology"

    def test_upsert_update(self, test_db):
        upsert_ticker(test_db, "AAPL", name="Apple Inc.", type="stock")
        upsert_ticker(test_db, "AAPL", name="Apple Inc. Updated", type="stock")
        t = get_ticker(test_db, "AAPL")
        assert t["name"] == "Apple Inc. Updated"

    def test_get_nonexistent(self, test_db):
        assert get_ticker(test_db, "FAKE") is None

    def test_list_tickers(self, test_db):
        upsert_ticker(test_db, "AAPL", name="Apple", type="stock")
        upsert_ticker(test_db, "MSFT", name="Microsoft", type="stock")
        tickers = list_tickers(test_db)
        assert len(tickers) == 2
        assert tickers[0]["symbol"] == "AAPL"  # Alphabetical

    def test_list_tickers_needs_review(self, test_db):
        upsert_ticker(test_db, "AAPL", name="Apple", type="stock", needs_review=False)
        upsert_ticker(test_db, "XYZ", name="Unknown", type="etf", needs_review=True)
        review = list_tickers(test_db, needs_review=True)
        assert len(review) == 1
        assert review[0]["symbol"] == "XYZ"

    def test_delete_ticker(self, test_db):
        upsert_ticker(test_db, "AAPL", name="Apple", type="stock")
        assert delete_ticker(test_db, "AAPL")
        assert get_ticker(test_db, "AAPL") is None

    def test_delete_nonexistent(self, test_db):
        assert not delete_ticker(test_db, "FAKE")

    def test_ticker_count(self, test_db):
        assert get_ticker_count(test_db) == 0
        upsert_ticker(test_db, "AAPL", name="Apple", type="stock")
        upsert_ticker(test_db, "MSFT", name="Microsoft", type="stock")
        assert get_ticker_count(test_db) == 2

    def test_boolean_flags(self, test_db):
        upsert_ticker(test_db, "GLD", name="SPDR Gold", type="etf", is_gold=True, is_hard_money=True)
        t = get_ticker(test_db, "GLD")
        assert t["is_gold"] == 1
        assert t["is_hard_money"] == 1
        assert t["is_crypto"] == 0


class TestAccountQueries:
    """Test account operations."""

    def test_upsert_and_list(self, test_db):
        upsert_account(test_db, "test_acct", "Test Account", "taxable")
        accounts = list_accounts(test_db)
        assert len(accounts) == 1
        assert accounts[0]["name"] == "Test Account"


class TestScoringQueries:
    """Test scoring run and score operations."""

    def test_insert_scoring_run(self, test_db):
        run_id = insert_scoring_run(test_db, "run-001", vix_current=18.5, vix_regime="NORMAL")
        assert run_id == "run-001"
        run = get_latest_scoring_run(test_db)
        assert run["vix_current"] == 18.5

    def test_insert_score(self, test_db):
        upsert_ticker(test_db, "AAPL", name="Apple", type="stock")
        insert_scoring_run(test_db, "run-001")
        score_id = insert_score(
            test_db, "run-001", "AAPL",
            dcs=72.5, dcs_signal="HIGH CONVICTION",
            mq=0.85, fq=0.70, tov=0.60, mr=0.75, vc=0.50,
        )
        assert score_id > 0

    def test_get_latest_scores(self, test_db):
        upsert_ticker(test_db, "AAPL", name="Apple", type="stock")
        upsert_ticker(test_db, "MSFT", name="Microsoft", type="stock")
        insert_scoring_run(test_db, "run-001")
        insert_score(test_db, "run-001", "AAPL", dcs=72.5, dcs_signal="HIGH CONVICTION")
        insert_score(test_db, "run-001", "MSFT", dcs=55.0, dcs_signal="WATCH")

        scores = get_latest_scores(test_db)
        assert "AAPL" in scores
        assert scores["AAPL"]["dcs"] == 72.5

    def test_score_history(self, test_db):
        upsert_ticker(test_db, "AAPL", name="Apple", type="stock")
        insert_scoring_run(test_db, "run-001")
        insert_scoring_run(test_db, "run-002")
        insert_score(test_db, "run-001", "AAPL", dcs=65.0, dcs_signal="BUY DIP")
        insert_score(test_db, "run-002", "AAPL", dcs=72.0, dcs_signal="HIGH CONVICTION")

        history = get_score_history(test_db, "AAPL")
        assert len(history) == 2

    def test_update_scoring_run(self, test_db):
        insert_scoring_run(test_db, "run-001")
        update_scoring_run(test_db, "run-001", status="success", tickers_scored=50)
        run = get_latest_scoring_run(test_db)
        assert run["status"] == "success"
        assert run["tickers_scored"] == 50

    def test_list_scoring_runs(self, test_db):
        insert_scoring_run(test_db, "run-001")
        insert_scoring_run(test_db, "run-002")
        runs = list_scoring_runs(test_db)
        assert len(runs) == 2


class TestSignalQueries:
    """Test signal operations."""

    def test_insert_signal(self, test_db):
        upsert_ticker(test_db, "AAPL", name="Apple", type="stock")
        insert_scoring_run(test_db, "run-001")
        score_id = insert_score(test_db, "run-001", "AAPL", dcs=40.0, dcs_signal="WEAK")
        sig_id = insert_signal(
            test_db, score_id,
            signal_type="SELL_HARD",
            severity="CRITICAL",
            criterion="sma_breach",
            message="AAPL below 200d SMA for 10+ days",
        )
        assert sig_id > 0


class TestDrawdownQueries:
    """Test drawdown classification operations."""

    def test_upsert_and_get(self, test_db):
        upsert_ticker(test_db, "GLD", name="Gold", type="etf")
        upsert_drawdown_classification(
            test_db, "2026-02-15", "GLD", "HEDGE",
            downside_capture=-0.85, win_rate_in_dd=0.62,
        )
        classifications = get_drawdown_classifications(test_db)
        assert "GLD" in classifications
        assert classifications["GLD"]["classification"] == "HEDGE"


class TestFreshnessQueries:
    """Test data freshness tracking."""

    def test_update_and_get(self, test_db):
        update_data_freshness(test_db, "yfinance", "ok")
        freshness = get_data_freshness(test_db)
        assert "yfinance" in freshness
        assert freshness["yfinance"]["last_status"] == "ok"
