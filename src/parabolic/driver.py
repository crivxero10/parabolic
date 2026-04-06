import argparse
import json
import logging
import os
import sys
from pathlib import Path

from parabolic.brokerage import Brokerage
from parabolic.classifier import RegimeClassifier, RegimeClassifierConfig
from parabolic.mdp import AlpacaMarketDataProvider, CachedMarketDataProvider
from parabolic.orchestrator import ContextOrchestrator
from parabolic.tuner import AdaptiveSearchConfig, ParameterRange, Tuner

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

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

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
    def strategy(ctx):
        classifier.apply_strategy(ctx)

    return strategy


# Inserted functions
def make_regime_parameters_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "k_st": int(args.k_st),
        "k_lt": int(args.k_lt),
        "lookback": int(args.lookback),
        "crab_lower_bound": float(args.crab_lower_bound),
        "crab_upper_bound": float(args.crab_upper_bound),
        "rolling_stop_pct": float(args.rolling_stop_pct),
    }


def resolve_strategy_factory(strategy_name: str):
    if strategy_name == "regime_classifier":
        return make_regime_strategy_factory()
    raise ValueError(f"Unsupported strategy name: {strategy_name}")


def build_regime_parameter_space() -> list[ParameterRange]:
    return [
        ParameterRange("k_st", (6, 8)),
        ParameterRange("k_lt", (42, 66)),
        ParameterRange("lookback", (11, 15)),
        ParameterRange("crab_lower_bound", (-3.0, -2.0, -1.5)),
        ParameterRange("crab_upper_bound", (1.5, 2.0, 3.0)),
        ParameterRange("rolling_stop_pct", (-0.10, -0.05)),
    ]


def make_regime_strategy_factory():
    def strategy_factory(parameters: dict[str, object]):
        config = RegimeClassifierConfig(
            k_st=int(parameters["k_st"]),
            k_lt=int(parameters["k_lt"]),
            lookback=int(parameters["lookback"]),
            crab_lower_bound=float(parameters["crab_lower_bound"]),
            crab_upper_bound=float(parameters["crab_upper_bound"]),
            rolling_stop_pct=float(parameters["rolling_stop_pct"]),
        )
        classifier = RegimeClassifier(config)
        return build_regime_strategy(classifier)

    return strategy_factory


def gather_market_data(
    provider: CachedMarketDataProvider,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    adjustment: str,
    feed: str | None,
) -> ContextOrchestrator:
    logger = logging.getLogger(__name__)

    logger.info("step=fetch_primary_bars symbol=%s timeframe=%s", symbol, timeframe)
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
    logger.info(
        "step=fetched_primary_bars symbol=%s bars=%s first_t=%s first_c=%s last_t=%s last_c=%s",
        symbol,
        len(bars),
        first_bar.get("t"),
        first_bar.get("c"),
        last_bar.get("t"),
        last_bar.get("c"),
    )

    logger.info("step=fetch_companion_bars symbol=SPXL timeframe=%s", timeframe)
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

    logger.info("step=fetch_companion_bars symbol=SPXS timeframe=%s", timeframe)
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

    logger.info("step=align_market_data primary=%s companions=SPXL,SPXS", symbol)
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
        raise ValueError("Finished alignment: unable to align market data on timestamps.")

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

    logger.info("step=finished_market_data_gathering aligned_snapshots=%s", len(snapshots))
    return orchestrator


def resolve_credentials(args: argparse.Namespace) -> tuple[str, str] | None:
    api_key = args.api_key or os.getenv("ALPACA_API_KEY")
    api_secret = args.api_secret or os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        return None
    return api_key, api_secret


def build_market_data_provider(args: argparse.Namespace, api_key: str, api_secret: str) -> CachedMarketDataProvider:
    alpaca_provider = AlpacaMarketDataProvider(api_key=api_key, api_secret=api_secret)
    return CachedMarketDataProvider(
        cache_dir=args.cache_dir,
        alpaca_provider=alpaca_provider,
    )


def run_regime_tuning(
    *,
    symbol: str,
    orchestrator: ContextOrchestrator,
    initial_balance: float,
) -> dict[str, dict[str, object]]:
    parameter_space = build_regime_parameter_space()
    strategy_factory = make_regime_strategy_factory()
    search_config = AdaptiveSearchConfig(
        initial_samples=12,
        refinement_rounds=2,
        top_k=3,
        random_seed=42,
    )

    def brokerage_factory() -> Brokerage:
        return Brokerage(balance=initial_balance)

    tuner = Tuner(
        asset_name=symbol,
        context_orchestrator=orchestrator,
        brokerage_factory=brokerage_factory,
        initial_balance=initial_balance,
        search_config=search_config,
    )
    results = tuner.adaptive_search(
        strategy_name="regime_classifier",
        parameter_space=parameter_space,
        strategy_factory=strategy_factory,
    )
    if not results:
        raise ValueError("No optimization results were produced.")
    return Tuner.summarize_best_results(results)


# Inserted function after run_regime_tuning
def evaluate_strategy(
    *,
    symbol: str,
    orchestrator: ContextOrchestrator,
    initial_balance: float,
    strategy_name: str,
    parameters: dict[str, object],
) -> dict[str, object]:
    strategy_factory = resolve_strategy_factory(strategy_name)

    def brokerage_factory() -> Brokerage:
        return Brokerage(balance=initial_balance)

    tuner = Tuner(
        asset_name=symbol,
        context_orchestrator=orchestrator,
        brokerage_factory=brokerage_factory,
        initial_balance=initial_balance,
        search_config=AdaptiveSearchConfig(),
    )
    result = tuner.evaluate(
        strategy_name=strategy_name,
        parameters=parameters,
        strategy_factory=strategy_factory,
    )
    if result is None:
        raise ValueError("Strategy evaluation did not produce a result.")
    return {
        "strategy_name": result.strategy_name,
        "parameters": result.parameters,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "calmar": result.calmar,
        "max_drawdown": result.max_drawdown,
        "final_balance": result.final_balance,
    }



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a backtest from Alpaca data and either tune or evaluate a strategy.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. SPY")
    common_parser.add_argument(
        "--start",
        required=True,
        help="Start datetime/date, e.g. 2024-01-01 or 2024-01-01T00:00:00Z",
    )
    common_parser.add_argument(
        "--end",
        required=True,
        help="End datetime/date, e.g. 2024-01-31 or 2024-01-31T00:00:00Z",
    )
    common_parser.add_argument(
        "--timeframe",
        default="minute",
        choices=sorted(TIMEFRAME_MAP.keys()),
        help="Bar timeframe to fetch",
    )
    common_parser.add_argument(
        "--adjustment",
        default="all",
        help="Alpaca adjustment mode, e.g. all, raw, split, dividend",
    )
    common_parser.add_argument("--feed", default=None, help="Optional Alpaca feed value")
    common_parser.add_argument(
        "--cache-dir",
        default=".cache/market_data",
        help="Directory containing the market data cache files",
    )
    common_parser.add_argument(
        "--api-key",
        default=None,
        help="Alpaca API key. Falls back to ALPACA_API_KEY env var.",
    )
    common_parser.add_argument(
        "--api-secret",
        default=None,
        help="Alpaca API secret. Falls back to ALPACA_API_SECRET env var.",
    )
    common_parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Starting brokerage balance for each simulated run",
    )

    subparsers.add_parser(
        "tune",
        parents=[common_parser],
        help="Tune the configuration of a strategy over the requested date range",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        parents=[common_parser],
        help="Evaluate a named strategy with an explicit configuration over the requested date range",
    )
    evaluate_parser.add_argument(
        "--strategy-name",
        default="regime_classifier",
        choices=["regime_classifier"],
        help="Strategy name to evaluate",
    )
    evaluate_parser.add_argument("--k-st", type=int, required=True, help="Short-term window")
    evaluate_parser.add_argument("--k-lt", type=int, required=True, help="Long-term window")
    evaluate_parser.add_argument("--lookback", type=int, required=True, help="Lookback window")
    evaluate_parser.add_argument(
        "--crab-lower-bound",
        type=float,
        required=True,
        help="Lower crab regime bound",
    )
    evaluate_parser.add_argument(
        "--crab-upper-bound",
        type=float,
        required=True,
        help="Upper crab regime bound",
    )
    evaluate_parser.add_argument(
        "--rolling-stop-pct",
        type=float,
        required=True,
        help="Rolling stop percentage",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("step=resolve_credentials")
    credentials = resolve_credentials(args)
    if credentials is None:
        print(
            "Missing Alpaca credentials. Set --api-key/--api-secret or ALPACA_API_KEY/ALPACA_API_SECRET.",
            file=sys.stderr,
        )
        return 1
    api_key, api_secret = credentials

    logger.info("step=build_market_data_provider")
    provider = build_market_data_provider(args, api_key, api_secret)

    logger.info("step=gather_market_data")
    try:
        orchestrator = gather_market_data(
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

    if args.command == "tune":
        logger.info("step=run_regime_tuning")
        try:
            summary = run_regime_tuning(
                symbol=args.symbol,
                orchestrator=orchestrator,
                initial_balance=args.initial_balance,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    elif args.command == "evaluate":
        logger.info("step=evaluate_strategy")
        try:
            summary = evaluate_strategy(
                symbol=args.symbol,
                orchestrator=orchestrator,
                initial_balance=args.initial_balance,
                strategy_name=args.strategy_name,
                parameters=make_regime_parameters_from_args(args),
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    else:
        print(f"Unsupported command: {args.command}", file=sys.stderr)
        return 1

    logger.info("step=emit_summary")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())