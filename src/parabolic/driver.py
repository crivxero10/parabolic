import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from parabolic.backtest import Backtester
from parabolic.brokerage import Brokerage
from parabolic.classifier import RegimeClassifier, RegimeClassifierConfig
from parabolic.mdp import AlpacaMarketDataProvider, CachedMarketDataProvider
from parabolic.orchestrator import ContextOrchestrator
from parabolic.strategy_runtime import StrategyRuntimeConfig, evaluate_strategy_source
from parabolic.tuner import (
    AdaptiveSearchConfig,
    ParameterRange,
    Tuner,
    returns_from_equity_curve,
    summarize_risk_metrics,
)

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

DEFAULT_COMPANION_SYMBOLS = ("SPXL", "SPXS")


def configure_logging(config_path: Path = LOGGING_CONFIG_PATH) -> dict:
    if not config_path.exists():
        config_path.write_text(
            json.dumps(DEFAULT_LOGGING_CONFIG, indent=2),
            encoding="utf-8",
        )

    config = DEFAULT_LOGGING_CONFIG | json.loads(config_path.read_text(encoding="utf-8"))
    level_name = str(config.get("level", "DEBUG")).upper()
    level = getattr(logging, level_name, logging.DEBUG)
    destination = str(config.get("destination", "file"))
    configured_file_path = Path(str(config.get("file_path", ".cache/parabolic.log"))).expanduser()
    if not configured_file_path.is_absolute():
        configured_file_path = (Path.cwd() / configured_file_path).resolve()
    file_path = _build_run_log_file_path(configured_file_path)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if destination == "terminal":
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    root_logger.propagate = False

    logging.getLogger(__name__).info(
        "logging_configured destination=%s level=%s configured_file_path=%s file_path=%s",
        destination,
        level_name,
        configured_file_path,
        file_path,
    )
    return config | {"file_path": file_path}


def _build_run_log_file_path(configured_file_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    suffix = "".join(configured_file_path.suffixes)
    base_name = configured_file_path.name
    if suffix:
        base_name = base_name[: -len(suffix)]
    run_file_name = f"{base_name}-{timestamp}-{os.getpid()}{suffix or '.log'}"
    return configured_file_path.with_name(run_file_name)


def build_regime_strategy(classifier: RegimeClassifier):
    def strategy(ctx):
        classifier.apply_strategy(ctx)

    return strategy


def strategy_api_spec() -> dict[str, object]:
    return {
        "strategy_signature": "def strategy(ctx): ...",
        "strategy_modes": {
            "builtin": "--strategy-name",
            "stdin": "--strategy-stdin",
        },
        "ctx_fields": {
            "t": "Current bar index within the session window",
            "asset_name": "Primary symbol passed with --symbol",
            "market": "List of market snapshots up to and including ctx.t",
            "bar": "Current raw bar dict for the primary symbol",
            "bars": "Raw primary-symbol bars up to and including ctx.t",
            "start_date": "Requested start date",
            "end_date": "Requested end date",
            "timeframe": "Requested timeframe",
            "adjustment": "Requested adjustment mode",
            "feed": "Requested feed",
            "session_market": "Entire session snapshot list",
            "session_length": "Total number of snapshots in the session",
            "is_session_start": "True on the first bar",
            "is_session_end": "True on the last bar",
            "brokerage": "Brokerage instance for order placement and portfolio state",
        },
        "brokerage_methods": {
            "execute": "execute(asset_name, units, price, timestamp=None)",
            "defer": "defer(asset_name, units, target_price, activate, position_id=None)",
            "liquidate": "liquidate(market_snapshot)",
            "get_total_pnl": "get_total_pnl(market_snapshot)",
            "get_avg_cost_basis": "get_avg_cost_basis(asset_name)",
        },
        "brokerage_fields": {
            "balance": "Realized cash balance",
            "available_cash": "Spendable cash",
            "positions": "Mapping of asset symbol to units held",
            "operations": "Executed operation history",
            "deferred_instructions": "Pending deferred instructions",
        },
        "indicator_methods": {
            "availability": "Indicators is injected into stdin strategy execution and can be used without import",
            "sma": "sma(series, n)",
            "ema": "ema(series, alpha)",
            "ema_window": "ema_window(n, series)",
            "ema_area_between_curves": "ema_area_between_curves(series, k_st, k_lt, lookback)",
            "rolling_std": "rolling_std(series, n)",
            "bollinger_bands": "bollinger_bands(series, n, num_std=2.0)",
            "rsi": "rsi(series, n=14)",
            "true_range": "true_range(highs, lows, closes)",
            "atr": "atr(highs, lows, closes, n=14)",
            "vwap": "vwap(highs, lows, closes, volumes)",
            "macd": "macd(series, fast=12, slow=26, signal=9)",
            "stochastic_oscillator": "stochastic_oscillator(highs, lows, closes, k_period=14, d_period=3)",
            "williams_r": "williams_r(highs, lows, closes, n=14)",
            "cci": "cci(highs, lows, closes, n=20)",
            "mfi": "mfi(highs, lows, closes, volumes, n=14)",
            "obv": "obv(closes, volumes)",
        },
        "market_symbols": {
            "default_behavior": "If --market-symbols is omitted, the market universe defaults to [--symbol, SPXL, SPXS]",
            "override_flag": "--market-symbols",
        },
        "stdin_guardrails": {
            "required_shape": "Exactly one top-level function named strategy with signature strategy(ctx)",
            "disallowed": [
                "imports",
                "while loops",
                "try/except blocks",
                "with blocks",
                "dunder attribute access",
                "dangerous builtin calls",
            ],
            "timeout_flag": "--strategy-timeout-seconds",
        },
    }


def parse_market_symbols(symbol: str, market_symbols_arg: str | None) -> list[str]:
    if market_symbols_arg is None:
        requested = [symbol, *DEFAULT_COMPANION_SYMBOLS]
    else:
        requested = [part.strip().upper() for part in market_symbols_arg.split(",") if part.strip()]
        if not requested:
            raise ValueError("--market-symbols must contain at least one symbol")
        if symbol not in requested:
            requested.insert(0, symbol)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in requested:
        normalized = item.upper()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


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


def make_regime_parameters_from_optional_args(args: argparse.Namespace) -> dict[str, object]:
    required = (
        "k_st",
        "k_lt",
        "lookback",
        "crab_lower_bound",
        "crab_upper_bound",
        "rolling_stop_pct",
    )
    missing = [name for name in required if getattr(args, name, None) is None]
    if missing:
        normalized = [f"--{name.replace('_', '-')}" for name in missing]
        raise ValueError(f"Missing required evaluate arguments for built-in strategy mode: {', '.join(normalized)}")
    return make_regime_parameters_from_args(args)


def resolve_strategy_factory(strategy_name: str):
    if strategy_name == "regime_classifier":
        return make_regime_strategy_factory()
    if strategy_name == "deterministic_test":
        return make_deterministic_test_strategy_factory()
    if strategy_name == "deterministic_tune":
        return make_deterministic_tune_strategy_factory()
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


def make_deterministic_test_strategy_factory():
    def strategy_factory(parameters: dict[str, object]):
        del parameters

        def strategy(ctx):
            current_price = float(ctx.market[ctx.t][ctx.asset_name])
            current_units = int(ctx.brokerage.positions.get(ctx.asset_name, 0))
            timestamp = None
            if getattr(ctx, "bar", None) is not None:
                timestamp = ctx.bar.get("t")

            if ctx.t == 1 and current_units == 0:
                units = min(10, math.floor(ctx.brokerage.available_cash / current_price))
                if units > 0:
                    ctx.brokerage.execute(
                        ctx.asset_name,
                        units,
                        current_price,
                        timestamp=timestamp,
                    )
                return

            if getattr(ctx, "is_session_end", False) and current_units > 0:
                ctx.brokerage.execute(
                    ctx.asset_name,
                    -current_units,
                    current_price,
                    timestamp=timestamp,
                )

        return strategy

    return strategy_factory


def build_deterministic_tune_parameter_space() -> list[ParameterRange]:
    return [
        ParameterRange("k_st", (5, 30, 60)),
        ParameterRange("k_lt", (80, 120)),
        ParameterRange("crab_upper_bound", (1.5, 2.0)),
    ]


def make_deterministic_tune_strategy_factory():
    def strategy_factory(parameters: dict[str, object]):
        entry_minute = int(parameters["k_st"])
        units_target = int(parameters["k_lt"])
        trade_asset = "SPXL" if float(parameters["crab_upper_bound"]) < 1.75 else "SPXS"

        def strategy(ctx):
            current_units = int(ctx.brokerage.positions.get(trade_asset, 0))
            current_price = float(ctx.market[ctx.t][trade_asset])
            timestamp = None
            if getattr(ctx, "bar", None) is not None:
                timestamp = ctx.bar.get("t")

            if ctx.t == entry_minute and current_units == 0:
                units = min(units_target, math.floor(ctx.brokerage.available_cash / current_price))
                if units > 0:
                    ctx.brokerage.execute(
                        trade_asset,
                        units,
                        current_price,
                        timestamp=timestamp,
                    )
                return

            if getattr(ctx, "is_session_end", False) and current_units > 0:
                ctx.brokerage.execute(
                    trade_asset,
                    -current_units,
                    current_price,
                    timestamp=timestamp,
                )

        return strategy

    return strategy_factory


def resolve_tuning_setup(strategy_name: str) -> tuple[list[ParameterRange], object, AdaptiveSearchConfig]:
    if strategy_name == "regime_classifier":
        return (
            build_regime_parameter_space(),
            make_regime_strategy_factory(),
            AdaptiveSearchConfig(
                initial_samples=12,
                refinement_rounds=2,
                top_k=3,
                random_seed=42,
            ),
        )

    if strategy_name == "deterministic_tune":
        return (
            build_deterministic_tune_parameter_space(),
            make_deterministic_tune_strategy_factory(),
            AdaptiveSearchConfig(
                initial_samples=12,
                refinement_rounds=0,
                top_k=3,
                random_seed=42,
                executor_kind="sequential",
                max_full_search_combinations=100,
                expand_numeric_parameters=False,
            ),
        )

    raise ValueError(f"Unsupported tuning strategy name: {strategy_name}")


def gather_market_data(
    provider: CachedMarketDataProvider,
    symbol: str,
    market_symbols: list[str],
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

    companion_symbols = [market_symbol for market_symbol in market_symbols if market_symbol != symbol]
    companion_prices_by_symbol: dict[str, dict[str, float]] = {}
    for companion_symbol in companion_symbols:
        logger.info("step=fetch_companion_bars symbol=%s timeframe=%s", companion_symbol, timeframe)
        companion_bars = provider.get_bars(
            symbol=companion_symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment=adjustment,
            feed=feed,
        )
        if not companion_bars:
            raise ValueError(f"Finished fetching {companion_symbol}: no data returned.")
        companion_prices_by_symbol[companion_symbol] = {
            bar["t"]: float(bar["c"])
            for bar in companion_bars
            if "t" in bar and "c" in bar
        }

    logger.info("step=align_market_data primary=%s companions=%s", symbol, ",".join(companion_symbols) or "none")

    snapshots: list[dict[str, float]] = []
    aligned_raw_bars: list[dict] = []
    for bar in bars:
        timestamp = bar.get("t")
        if timestamp is None:
            continue
        if any(timestamp not in prices_by_timestamp for prices_by_timestamp in companion_prices_by_symbol.values()):
            continue
        aligned_raw_bars.append(bar)
        snapshot = {symbol: float(bar["c"])}
        for companion_symbol in companion_symbols:
            snapshot[companion_symbol] = companion_prices_by_symbol[companion_symbol][timestamp]
        snapshots.append(snapshot)

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
    strategy_name: str,
    symbol: str,
    orchestrator: ContextOrchestrator,
    initial_balance: float,
) -> dict[str, dict[str, object]]:
    parameter_space, strategy_factory, search_config = resolve_tuning_setup(strategy_name)

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
        strategy_name=strategy_name,
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
    start: str,
    end: str,
    log_file: str,
) -> dict[str, object]:
    logger = logging.getLogger(__name__)
    strategy_factory = resolve_strategy_factory(strategy_name)
    strategy = strategy_factory(parameters)

    logger.info(
        "evaluation_artifact_logging mode=plain_text log_file=%s strategy_name=%s start=%s end=%s",
        log_file,
        strategy_name,
        start,
        end,
    )

    seed_brokerage = Brokerage(balance=initial_balance)
    backtester = Backtester(
        strategy=strategy,
        brokerage=seed_brokerage,
        asset_name=symbol,
        context_orchestrator=orchestrator,
        collect_equity_curve=False,
        collect_closed_trades=False,
        collect_daily_snapshots=True,
        collect_steps=False,
    )

    daily_results = backtester.simulate_by_day(
        brokerage=seed_brokerage,
        strategy=strategy,
        brokerage_factory=lambda: Brokerage(balance=initial_balance),
    )

    balances = [result.end_balance for result in daily_results]
    if not balances:
        raise ValueError("Strategy evaluation did not produce a result.")

    metric_equity_curve = [initial_balance, *[float(balance) for balance in balances]]
    returns = returns_from_equity_curve(metric_equity_curve)
    if not returns:
        raise ValueError("Strategy evaluation did not produce usable return series.")

    metrics = summarize_risk_metrics(
        returns=returns,
        equity_curve=metric_equity_curve,
    )

    daily_pnl_series = [
        {
            "date": result.session_date if result.session_date is not None else str(index + 1),
            "pnl_amount": result.daily_pnl_amount,
            "pnl_pct": result.daily_pnl_pct,
            "ending_balance": result.end_balance,
        }
        for index, result in enumerate(daily_results)
        if result.daily_pnl_pct is not None
    ] or None

    class EvaluationResult:
        def __init__(self):
            self.strategy_name = strategy_name
            self.parameters = dict(parameters)
            self.sharpe = metrics["sharpe"]
            self.sortino = metrics["sortino"]
            self.calmar = metrics["calmar"]
            self.max_drawdown = metrics["max_drawdown"]
            self.final_balance = balances[-1]
            self.initial_balance = initial_balance
            self.daily_pnl_series = daily_pnl_series

    summary = _build_evaluation_summary(EvaluationResult(), start=start, end=end)
    summary["log_file"] = log_file
    return summary


# Helper functions for evaluation summary
def _parse_date_like(value: str) -> date:
    normalized = value.strip()
    if "T" in normalized:
        normalized = normalized.split("T", 1)[0]
    return date.fromisoformat(normalized)


def _safe_average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _group_daily_series_to_weeks(daily_pnl_series: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    ordered_keys: list[tuple[int, int]] = []
    for row in daily_pnl_series:
        date_value = _parse_date_like(str(row["date"]))
        iso_year, iso_week, _ = date_value.isocalendar()
        key = (iso_year, iso_week)
        if key not in grouped:
            ordered_keys.append(key)
        grouped[key].append(row)
    return [grouped[key] for key in ordered_keys]


def _group_daily_series_to_months(daily_pnl_series: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    ordered_keys: list[tuple[int, int]] = []
    for row in daily_pnl_series:
        date_value = _parse_date_like(str(row["date"]))
        key = (date_value.year, date_value.month)
        if key not in grouped:
            ordered_keys.append(key)
        grouped[key].append(row)
    return [grouped[key] for key in ordered_keys]


def _group_daily_series_to_years(daily_pnl_series: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    ordered_keys: list[int] = []
    for row in daily_pnl_series:
        date_value = _parse_date_like(str(row["date"]))
        key = date_value.year
        if key not in grouped:
            ordered_keys.append(key)
        grouped[key].append(row)
    return [grouped[key] for key in ordered_keys]


def _period_is_longer_than_week(start: str, end: str) -> bool:
    return (_parse_date_like(end) - _parse_date_like(start)).days > 7


def _period_is_longer_than_month(start: str, end: str) -> bool:
    start_date = _parse_date_like(start)
    end_date = _parse_date_like(end)
    return (end_date.year, end_date.month) != (start_date.year, start_date.month) and (end_date - start_date).days > 31


def _period_is_longer_than_year(start: str, end: str) -> bool:
    start_date = _parse_date_like(start)
    end_date = _parse_date_like(end)
    return end_date.year > start_date.year and (end_date - start_date).days > 366


def _build_evaluation_summary(result, start: str, end: str) -> dict[str, object]:
    summary: dict[str, object] = {
        "strategy_name": result.strategy_name,
        "parameters": result.parameters,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "calmar": result.calmar,
        "max_drawdown": result.max_drawdown,
        "final_balance": result.final_balance,
    }

    if result.initial_balance is not None and result.initial_balance != 0:
        summary["pnl_pct_all_time"] = ((result.final_balance / result.initial_balance) - 1.0) * 100.0
    else:
        summary["pnl_pct_all_time"] = None

    daily_pnl_series = result.daily_pnl_series
    if daily_pnl_series:
        daily_pct_values = [
            float(row["pnl_pct"]) * 100.0
            for row in daily_pnl_series
            if row.get("pnl_pct") is not None
        ]
        summary["pnl_pct_daily_average"] = _safe_average(daily_pct_values)

        if _period_is_longer_than_week(start, end):
            weekly_groups = _group_daily_series_to_weeks(daily_pnl_series)
            weekly_values = [
                _safe_average([
                    float(row["pnl_pct"]) * 100.0
                    for row in group
                    if row.get("pnl_pct") is not None
                ])
                for group in weekly_groups
            ]
            summary["pnl_pct_weekly_average"] = _safe_average(
                [value for value in weekly_values if value is not None]
            )

        if _period_is_longer_than_month(start, end):
            monthly_groups = _group_daily_series_to_months(daily_pnl_series)
            monthly_values = [
                _safe_average([
                    float(row["pnl_pct"]) * 100.0
                    for row in group
                    if row.get("pnl_pct") is not None
                ])
                for group in monthly_groups
            ]
            summary["pnl_pct_monthly_average"] = _safe_average(
                [value for value in monthly_values if value is not None]
            )

        if _period_is_longer_than_year(start, end):
            yearly_groups = _group_daily_series_to_years(daily_pnl_series)
            yearly_values = [
                _safe_average([
                    float(row["pnl_pct"]) * 100.0
                    for row in group
                    if row.get("pnl_pct") is not None
                ])
                for group in yearly_groups
            ]
            summary["pnl_pct_yearly_average"] = _safe_average(
                [value for value in yearly_values if value is not None]
            )

        valid_day_rows = [row for row in daily_pnl_series if row.get("pnl_pct") is not None]
        if valid_day_rows:
            best_day = max(valid_day_rows, key=lambda row: float(row["pnl_pct"]))
            worst_day = min(valid_day_rows, key=lambda row: float(row["pnl_pct"]))
            summary["best_day"] = {
                "date": best_day["date"],
                "pnl_pct": float(best_day["pnl_pct"]) * 100.0,
            }
            summary["worst_day"] = {
                "date": worst_day["date"],
                "pnl_pct": float(worst_day["pnl_pct"]) * 100.0,
            }
        else:
            summary["best_day"] = None
            summary["worst_day"] = None
    else:
        summary["pnl_pct_daily_average"] = None
        summary["best_day"] = None
        summary["worst_day"] = None
        if _period_is_longer_than_week(start, end):
            summary["pnl_pct_weekly_average"] = None
        if _period_is_longer_than_month(start, end):
            summary["pnl_pct_monthly_average"] = None
        if _period_is_longer_than_year(start, end):
            summary["pnl_pct_yearly_average"] = None

    return summary



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a backtest from Alpaca data and either tune or evaluate a strategy.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. SPY")
    common_parser.add_argument(
        "--market-symbols",
        default=None,
        help="Comma-separated market universe available to the strategy. Defaults to the primary symbol plus SPXL and SPXS.",
    )
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
    tune_parser = subparsers.choices["tune"]
    tune_parser.add_argument(
        "--strategy-name",
        default="regime_classifier",
        choices=["regime_classifier", "deterministic_tune"],
        help="Strategy name to tune",
    )

    subparsers.add_parser(
        "strategy-spec",
        help="Emit a machine-readable JSON specification for strategy authors",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        parents=[common_parser],
        help="Evaluate a named strategy with an explicit configuration over the requested date range",
    )
    evaluate_parser.add_argument(
        "--strategy-name",
        default="regime_classifier",
        choices=["regime_classifier", "deterministic_test"],
        help="Strategy name to evaluate",
    )
    evaluate_parser.add_argument(
        "--strategy-stdin",
        action="store_true",
        help="Read a Python strategy from stdin. The submitted source must define exactly `def strategy(ctx): ...`",
    )
    evaluate_parser.add_argument(
        "--strategy-timeout-seconds",
        type=float,
        default=1000.0,
        help="Maximum runtime allowed for stdin strategy evaluation before the subprocess is terminated",
    )
    evaluate_parser.add_argument("--k-st", type=int, required=False, help="Short-term window")
    evaluate_parser.add_argument("--k-lt", type=int, required=False, help="Long-term window")
    evaluate_parser.add_argument("--lookback", type=int, required=False, help="Lookback window")
    evaluate_parser.add_argument(
        "--crab-lower-bound",
        type=float,
        required=False,
        help="Lower crab regime bound",
    )
    evaluate_parser.add_argument(
        "--crab-upper-bound",
        type=float,
        required=False,
        help="Upper crab regime bound",
    )
    evaluate_parser.add_argument(
        "--rolling-stop-pct",
        type=float,
        required=False,
        help="Rolling stop percentage",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    market_data_provider_override: CachedMarketDataProvider | None = None,
    stdin_override: str | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging_config = configure_logging()
    logger = logging.getLogger(__name__)

    if args.command == "strategy-spec":
        print(json.dumps(strategy_api_spec(), indent=2))
        return 0

    if market_data_provider_override is None:
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
    else:
        provider = market_data_provider_override

    try:
        market_symbols = parse_market_symbols(args.symbol, getattr(args, "market_symbols", None))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    logger.info("step=gather_market_data")
    try:
        orchestrator = gather_market_data(
            provider=provider,
            symbol=args.symbol,
            market_symbols=market_symbols,
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
                strategy_name=args.strategy_name,
                symbol=args.symbol,
                orchestrator=orchestrator,
                initial_balance=args.initial_balance,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    elif args.command == "evaluate":
        logger.info("step=evaluate_strategy")
        if args.strategy_stdin:
            strategy_source = stdin_override if stdin_override is not None else sys.stdin.read()
            success, payload, exit_code = evaluate_strategy_source(
                strategy_source=strategy_source,
                symbol=args.symbol,
                raw_bars=list(getattr(orchestrator, "raw_bars", [])),
                snapshots=orchestrator.get_snapshots(),
                start=args.start,
                end=args.end,
                initial_balance=args.initial_balance,
                log_file=str(logging_config["file_path"]),
                config=StrategyRuntimeConfig(timeout_seconds=float(args.strategy_timeout_seconds)),
            )
            print(json.dumps(payload, indent=2))
            return exit_code
        try:
            summary = evaluate_strategy(
                symbol=args.symbol,
                orchestrator=orchestrator,
                initial_balance=args.initial_balance,
                strategy_name=args.strategy_name,
                parameters=make_regime_parameters_from_optional_args(args),
                start=args.start,
                end=args.end,
                log_file=str(logging_config["file_path"]),
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
