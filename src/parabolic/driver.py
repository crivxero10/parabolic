import argparse
import json
import logging
import os
import sys
import math
from pathlib import Path
from parabolic.backtest import Backtester, TradingContext
from parabolic.brokerage import Brokerage
from parabolic.mdp import AlpacaMarketDataProvider, CachedMarketDataProvider
from parabolic.orchestrator import ContextOrchestrator
from parabolic.classifier import RegimeClassifier
from dataclasses import dataclass

TIMEFRAME_MAP = {
    "minute": "1Min",
    "daily": "1Day",
}

LOGGING_CONFIG_PATH = Path("logging_config.json")
DEFAULT_LOGGING_CONFIG = {
    "level": "DEBUG",
    "destination": "file",
    "file_path": ".cache/parabolic.log",
}


def configure_logging(config_path: Path = LOGGING_CONFIG_PATH) -> dict:
    if not config_path.exists():
        config_path.write_text(
            json.dumps(DEFAULT_LOGGING_CONFIG, indent=2),
            encoding="utf-8",
        )

    config = DEFAULT_LOGGING_CONFIG | json.loads(config_path.read_text(encoding="utf-8"))
    level_name = str(config.get("level", "DEBUG")).upper()
    level = getattr(logging, level_name, logging.DEBUG)
    destination = config.get("destination", "terminal")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    if destination == "file":
        file_path = Path(str(config.get("file_path", ".cache/parabolic.log")))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setLevel(level)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.propagate = False

    logging.getLogger(__name__).info(
        "logging_configured destination=%s level=%s",
        destination,
        level_name,
    )
    return config

def build_regime_strategy(classifier: RegimeClassifier):
    def strategy(ctx: TradingContext):
        classifier.apply_strategy(ctx)
    return strategy

def gather_market_data(
    provider: CachedMarketDataProvider,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    adjustment: str,
    feed: str | None,
) -> tuple[list[dict], list[dict[str, float]], ContextOrchestrator]:
    print(f"Fetching {symbol} bars...", file=sys.stderr)
    bars = provider.get_bars(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment=adjustment,
        feed=feed,
    )
    if not bars:
        raise ValueError(f"Finished fetching {symbol}: no data returned.")

    first_bar = bars[0]
    last_bar = bars[-1]
    print(
        (
            f"Finished fetching {symbol}: {len(bars)} bars. "
            f"First bar: t={first_bar.get('t')} c={first_bar.get('c')}. "
            f"Last bar: t={last_bar.get('t')} c={last_bar.get('c')}."
        ),
        file=sys.stderr,
    )

    print("Fetching SPXL bars...", file=sys.stderr)
    spxl_bars = provider.get_bars(
        symbol="SPXL",
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment=adjustment,
        feed=feed,
    )
    if not spxl_bars:
        raise ValueError("Finished fetching SPXL: no data returned.")
    print(f"Finished fetching SPXL: {len(spxl_bars)} bars.", file=sys.stderr)

    print("Fetching SPXS bars...", file=sys.stderr)
    spxs_bars = provider.get_bars(
        symbol="SPXS",
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment=adjustment,
        feed=feed,
    )
    if not spxs_bars:
        raise ValueError("Finished fetching SPXS: no data returned.")
    print(f"Finished fetching SPXS: {len(spxs_bars)} bars.", file=sys.stderr)

    print("Aligning SPY/SPXL/SPXS timestamps...", file=sys.stderr)
    spxl_by_timestamp = {bar["t"]: float(bar["c"]) for bar in spxl_bars if "t" in bar and "c" in bar}
    spxs_by_timestamp = {bar["t"]: float(bar["c"]) for bar in spxs_bars if "t" in bar and "c" in bar}

    snapshots: list[dict[str, float]] = []
    aligned_raw_bars: list[dict] = []
    for bar in bars:
        timestamp = bar.get("t")
        if timestamp not in spxl_by_timestamp or timestamp not in spxs_by_timestamp:
            continue
        aligned_raw_bars.append(bar)
        snapshots.append(
            {
                symbol: float(bar["c"]),
                "SPXL": spxl_by_timestamp[timestamp],
                "SPXS": spxs_by_timestamp[timestamp],
            }
        )

    if not snapshots:
        raise ValueError("Finished alignment: unable to align SPY/SPXL/SPXS bars on timestamps.")

    print(f"Finished alignment: {len(snapshots)} aligned snapshots.", file=sys.stderr)

    orchestrator = ContextOrchestrator(
        snapshots=snapshots,
        asset_name=symbol,
        start_date=start,
        end_date=end,
        timeframe=timeframe,
        adjustment=adjustment,
        feed=feed,
    )
    orchestrator.raw_bars = aligned_raw_bars
    orchestrator._loaded = True
    print("Finished market data gathering.", file=sys.stderr)
    return aligned_raw_bars, snapshots, orchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a backtest from Alpaca/TinyDB data and plot the end-of-day balance series.",
    )
    parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. SPY")
    parser.add_argument("--start", required=True, help="Start datetime/date, e.g. 2024-01-01 or 2024-01-01T00:00:00Z")
    parser.add_argument("--end", required=True, help="End datetime/date, e.g. 2024-01-31 or 2024-01-31T00:00:00Z")
    parser.add_argument(
        "--timeframe",
        default="daily",
        choices=sorted(TIMEFRAME_MAP.keys()),
        help="Bar timeframe to fetch",
    )
    parser.add_argument(
        "--adjustment",
        default="all",
        help="Alpaca adjustment mode, e.g. all, raw, split, dividend",
    )
    parser.add_argument("--feed", default=None, help="Optional Alpaca feed value")
    parser.add_argument(
        "--cache-dir",
        default=".cache/market_data",
        help="Directory containing the TinyDB cache file",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Alpaca API key. Falls back to ALPACA_API_KEY env var.",
    )
    parser.add_argument(
        "--api-secret",
        default=None,
        help="Alpaca API secret. Falls back to ALPACA_API_SECRET env var.",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Starting brokerage balance for each simulated day",
    )
    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging()

    api_key = args.api_key or os.getenv("ALPACA_API_KEY")
    api_secret = args.api_secret or os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        print(
            "Missing Alpaca credentials. Set --api-key/--api-secret or ALPACA_API_KEY/ALPACA_API_SECRET.",
            file=sys.stderr,
        )
        return 1

    alpaca_provider = AlpacaMarketDataProvider(api_key=api_key, api_secret=api_secret)
    provider = CachedMarketDataProvider(
        cache_dir=args.cache_dir,
        alpaca_provider=alpaca_provider,
    )

    def brokerage_factory() -> Brokerage:
        return Brokerage(balance=args.initial_balance)

    seed_brokerage = brokerage_factory()
    classifier = RegimeClassifier()
    strategy = build_regime_strategy(classifier)

    try:
        _, _, orchestrator = gather_market_data(
            provider=provider,
            symbol=args.symbol,
            timeframe=TIMEFRAME_MAP[args.timeframe],
            start=args.start,
            end=args.end,
            adjustment=args.adjustment,
            feed=args.feed,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    backtester = Backtester(
        strategy=strategy,
        brokerage=seed_brokerage,
        asset_name=args.symbol,
        context_orchestrator=orchestrator,
    )

    daily_results = backtester.simulate_by_day(
        brokerage=seed_brokerage,
        strategy=strategy,
        brokerage_factory=brokerage_factory,
    )

    balances = [result.end_balance for result in daily_results]
    labels = [
        result.session_date if result.session_date is not None else str(index + 1)
        for index, result in enumerate(daily_results)
    ]

    if not balances:
        print("No data returned for the requested range.", file=sys.stderr)
        return 1
    print(balances)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())