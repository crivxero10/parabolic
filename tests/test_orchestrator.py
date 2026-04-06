import unittest
import parabolic.orchestrator as orchestrator_module
from parabolic.orchestrator import ContextOrchestrator


class TestOrchestrator(unittest.TestCase):

    @staticmethod
    def _make_orchestrator(raw_bars=None):
        orchestrator = ContextOrchestrator.__new__(ContextOrchestrator)
        orchestrator.market_data_provider = None
        orchestrator.asset_name = "SPY"
        orchestrator.start_date = None
        orchestrator.end_date = None
        orchestrator.timeframe = "1Min"
        orchestrator.adjustment = "all"
        orchestrator.feed = None
        orchestrator.context_factory = None
        orchestrator.extra_context = {}
        orchestrator.raw_bars = list(raw_bars or [])
        orchestrator._snapshots = [
            {"SPY": float(index + 1)}
            for index in range(len(orchestrator.raw_bars))
        ]
        orchestrator._loaded = True
        orchestrator._load_market_data = lambda: None
        return orchestrator

    def test_normalize_timestamp_handles_none_whitespace_and_existing_values(self):
        # Useful because callers may receive timestamps from raw bars in mixed string forms.
        # This should fail if whitespace is not stripped or if None is coerced into the string "None".
        assert ContextOrchestrator._normalize_timestamp(None) is None
        assert ContextOrchestrator._normalize_timestamp("") is None
        assert ContextOrchestrator._normalize_timestamp("   ") is None
        assert ContextOrchestrator._normalize_timestamp(" 2025-01-02T15:59:00Z ") == "2025-01-02T15:59:00Z"
        assert ContextOrchestrator._normalize_timestamp("2025-01-02") == "2025-01-02"

    def test_extract_trading_date_handles_iso_and_date_like_inputs(self):
        # Useful because daily grouping depends on extracting stable date keys from timestamps.
        # This should fail if the helper slices the wrong portion or mishandles date-only strings.
        assert ContextOrchestrator._extract_trading_date(None) is None
        assert ContextOrchestrator._extract_trading_date("2025-01-02T15:59:00Z") == "2025-01-02"
        assert ContextOrchestrator._extract_trading_date("2025-01-02") == "2025-01-02"
        assert ContextOrchestrator._extract_trading_date("20250102") == "20250102"

    def test_get_timestamp_rows_normalizes_rows_in_order_and_preserves_missing_timestamps(self):
        # Useful because evaluation/reporting needs a stable row-per-bar mapping.
        # This should fail if the method drops rows, reorders them, or fails to normalize whitespace.
        orchestrator = self._make_orchestrator(
            raw_bars=[
                {"t": " 2025-01-02T15:59:00Z "},
                {"t": None},
                {"t": "2025-01-03T15:58:00Z"},
                {"t": "2025-01-03"},
            ]
        )

        rows = orchestrator.get_timestamp_rows()

        assert rows == [
            {"timestamp": "2025-01-02T15:59:00Z", "date": "2025-01-02"},
            {"timestamp": None, "date": None},
            {"timestamp": "2025-01-03T15:58:00Z", "date": "2025-01-03"},
            {"timestamp": "2025-01-03", "date": "2025-01-03"},
        ]

    def test_get_trading_dates_deduplicates_in_order_and_skips_missing_dates(self):
        # Useful because downstream daily reports care about unique trading dates in sequence.
        # This should fail if duplicates are not removed, order changes, or None dates leak through.
        orchestrator = self._make_orchestrator()
        orchestrator.get_timestamp_rows = lambda: [
            {"timestamp": "2025-01-02T15:59:00Z", "date": "2025-01-02"},
            {"timestamp": None, "date": None},
            {"timestamp": "2025-01-02T16:00:00Z", "date": "2025-01-02"},
            {"timestamp": "2025-01-03T09:30:00Z", "date": "2025-01-03"},
            {"timestamp": "2025-01-01T15:59:00Z", "date": "2025-01-01"},
        ]

        dates = orchestrator.get_trading_dates()

        assert dates == ["2025-01-02", "2025-01-03", "2025-01-01"]

    def test_split_into_daily_orchestrators_groups_by_normalized_date_and_skips_missing(self):
        # Useful because this is the main consumer-facing daily partitioning behavior.
        # This should fail if malformed timestamps are included, rows are assigned to the wrong day,
        # or normalized timestamps do not produce the right grouping boundaries.
        orchestrator = self._make_orchestrator(
            raw_bars=[
                {"t": " 2025-01-02T15:58:00Z "},
                {"t": "2025-01-02T15:59:00Z"},
                {"t": None},
                {"t": "2025-01-03T15:58:00Z"},
            ]
        )

        daily_orchestrators = orchestrator.split_into_daily_orchestrators()

        assert len(daily_orchestrators) == 2
        assert [session_date for session_date, _ in daily_orchestrators] == [
            "2025-01-02",
            "2025-01-03",
        ]
        assert [daily.get_trading_dates() for _, daily in daily_orchestrators] == [
            ["2025-01-02"],
            ["2025-01-03"],
        ]
        assert [daily.get_timestamp_rows() for _, daily in daily_orchestrators] == [
            [
                {"timestamp": "2025-01-02T15:58:00Z", "date": "2025-01-02"},
                {"timestamp": "2025-01-02T15:59:00Z", "date": "2025-01-02"},
            ],
            [
                {"timestamp": None, "date": None},
                {"timestamp": "2025-01-03T15:58:00Z", "date": "2025-01-03"},
            ],
        ]
        assert [daily.get_snapshots() for _, daily in daily_orchestrators] == [
            [{"SPY": 1.0}, {"SPY": 2.0}],
            [{"SPY": 3.0}, {"SPY": 4.0}],
        ]

    def test_split_into_daily_orchestrators_returns_empty_for_only_missing_timestamps(self):
        # Useful edge case: bad timestamp payloads should not create phantom daily sessions.
        # This should fail if the grouping function creates empty or invalid daily orchestrators.
        orchestrator = self._make_orchestrator(
            raw_bars=[
                {"t": None},
                {"t": "   "},
            ]
        )

        daily_orchestrators = orchestrator.split_into_daily_orchestrators()

        assert daily_orchestrators == []

    def test_build_context_exposes_market_and_bar_history_for_strategies(self):
        orchestrator = self._make_orchestrator(
            raw_bars=[
                {"t": "2025-01-02T14:30:00Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000},
                {"t": "2025-01-02T14:31:00Z", "o": 100.5, "h": 102.0, "l": 100.0, "c": 101.5, "v": 1100},
                {"t": "2025-01-02T14:32:00Z", "o": 101.5, "h": 103.0, "l": 101.0, "c": 102.5, "v": 1200},
            ]
        )
        orchestrator._snapshots = [
            {"SPY": 100.5},
            {"SPY": 101.5},
            {"SPY": 102.5},
        ]

        ctx = orchestrator.build_context(2)

        assert ctx.market == [{"SPY": 100.5}, {"SPY": 101.5}, {"SPY": 102.5}]
        assert ctx.bar == {"t": "2025-01-02T14:32:00Z", "o": 101.5, "h": 103.0, "l": 101.0, "c": 102.5, "v": 1200}
        assert ctx.bars == [
            {"t": "2025-01-02T14:30:00Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000},
            {"t": "2025-01-02T14:31:00Z", "o": 100.5, "h": 102.0, "l": 100.0, "c": 101.5, "v": 1100},
            {"t": "2025-01-02T14:32:00Z", "o": 101.5, "h": 103.0, "l": 101.0, "c": 102.5, "v": 1200},
        ]
        assert ctx.is_session_start is False
        assert ctx.is_session_end is True
