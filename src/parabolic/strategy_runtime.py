from __future__ import annotations

import ast
import io
import multiprocessing
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any, Callable

from parabolic.backtest import Backtester
from parabolic.brokerage import Brokerage
from parabolic.indicators import Indicators
from parabolic.orchestrator import ContextOrchestrator
from parabolic.risk import returns_from_equity_curve, summarize_risk_metrics


SAFE_BUILTINS: dict[str, object] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "str": str,
    "sum": sum,
}

BANNED_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "delattr",
    "vars",
}

BANNED_AST_NODES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Try,
    ast.While,
    ast.With,
    ast.AsyncWith,
)


class StrategyValidationError(ValueError):
    pass


@dataclass(slots=True)
class StrategyRuntimeConfig:
    timeout_seconds: float = 1000.0
    max_captured_output_chars: int = 4000


def validate_strategy_source(source: str) -> None:
    try:
        module = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise StrategyValidationError(f"Strategy source has invalid syntax: {exc.msg} at line {exc.lineno}") from exc

    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(function_defs) != 1:
        raise StrategyValidationError("Strategy source must define exactly one top-level function")

    strategy_def = function_defs[0]
    if strategy_def.name != "strategy":
        raise StrategyValidationError("Strategy source must define `def strategy(ctx): ...`")

    if strategy_def.decorator_list:
        raise StrategyValidationError("Strategy function decorators are not allowed")

    if (
        len(strategy_def.args.args) != 1
        or strategy_def.args.args[0].arg != "ctx"
        or strategy_def.args.vararg is not None
        or strategy_def.args.kwarg is not None
        or strategy_def.args.kwonlyargs
        or strategy_def.args.posonlyargs
    ):
        raise StrategyValidationError("Strategy signature must be exactly `def strategy(ctx): ...`")

    for node in module.body:
        if node is strategy_def:
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        raise StrategyValidationError("Only a module docstring and `def strategy(ctx): ...` are allowed")

    for node in ast.walk(strategy_def):
        if isinstance(node, BANNED_AST_NODES):
            raise StrategyValidationError(f"Disallowed construct in strategy source: {type(node).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            raise StrategyValidationError(f"Disallowed function call in strategy source: {node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise StrategyValidationError("Dunder attribute access is not allowed in strategy source")


def load_strategy_from_source(source: str) -> Callable[[Any], None]:
    validate_strategy_source(source)
    namespace: dict[str, object] = {"__builtins__": SAFE_BUILTINS, "Indicators": Indicators}
    locals_namespace: dict[str, object] = {}
    exec(compile(source, "<strategy-stdin>", "exec"), namespace, locals_namespace)
    strategy = locals_namespace.get("strategy", namespace.get("strategy"))
    if not callable(strategy):
        raise StrategyValidationError("Strategy source did not define a callable `strategy`")
    return strategy


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...<truncated>"


def _build_error_payload(
    *,
    error_type: str,
    message: str,
    phase: str,
    max_output_chars: int,
    stdout: str = "",
    stderr: str = "",
    tb: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error_type": error_type,
        "message": message,
        "phase": phase,
    }
    if tb:
        payload["traceback"] = _truncate(tb, max_output_chars)
    if stdout:
        payload["captured_stdout"] = _truncate(stdout, max_output_chars)
    if stderr:
        payload["captured_stderr"] = _truncate(stderr, max_output_chars)
    if context:
        payload["context"] = context
    return payload


def _build_daily_pnl_series(
    *,
    daily_results: list[Any],
) -> list[dict[str, object]] | None:
    rows = [
        {
            "date": result.session_date if result.session_date is not None else str(index + 1),
            "pnl_amount": result.daily_pnl_amount,
            "pnl_pct": result.daily_pnl_pct,
            "ending_balance": result.end_balance,
        }
        for index, result in enumerate(daily_results)
        if result.daily_pnl_pct is not None
    ]
    return rows or None


def _evaluate_strategy_callable(
    *,
    strategy: Callable[[Any], None],
    symbol: str,
    raw_bars: list[dict[str, Any]],
    snapshots: list[dict[str, float]],
    start: str,
    end: str,
    initial_balance: float,
    log_file: str,
) -> dict[str, object]:
    from parabolic.driver import _build_evaluation_summary

    orchestrator = ContextOrchestrator(
        snapshots=snapshots,
        asset_name=symbol,
        start_date=start,
        end_date=end,
        timeframe="1Min",
    )
    orchestrator.raw_bars = [dict(bar) for bar in raw_bars]
    orchestrator._loaded = True

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

    daily_pnl_series = _build_daily_pnl_series(daily_results=daily_results)

    class EvaluationResult:
        def __init__(self):
            self.strategy_name = "strategy_stdin"
            self.parameters = {}
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


def _strategy_child_main(
    queue: multiprocessing.queues.Queue,
    *,
    strategy_source: str,
    symbol: str,
    raw_bars: list[dict[str, Any]],
    snapshots: list[dict[str, float]],
    start: str,
    end: str,
    initial_balance: float,
    log_file: str,
    max_output_chars: int,
) -> None:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    context: dict[str, Any] = {}
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            user_strategy = load_strategy_from_source(strategy_source)

            def wrapped_strategy(ctx):
                context["t"] = ctx.t
                context["asset_name"] = ctx.asset_name
                if getattr(ctx, "bar", None) is not None:
                    context["timestamp"] = ctx.bar.get("t")
                return user_strategy(ctx)

            summary = _evaluate_strategy_callable(
                strategy=wrapped_strategy,
                symbol=symbol,
                raw_bars=raw_bars,
                snapshots=snapshots,
                start=start,
                end=end,
                initial_balance=initial_balance,
                log_file=log_file,
            )
        queue.put({"ok": True, "summary": summary})
    except Exception as exc:
        queue.put(
            {
                "ok": False,
                "error": _build_error_payload(
                    error_type=type(exc).__name__,
                    message=str(exc),
                    phase="strategy_runtime",
                    max_output_chars=max_output_chars,
                    stdout=stdout_buffer.getvalue(),
                    stderr=stderr_buffer.getvalue(),
                    tb=traceback.format_exc(),
                    context=context or None,
                ),
            }
        )


def evaluate_strategy_source(
    *,
    strategy_source: str,
    symbol: str,
    raw_bars: list[dict[str, Any]],
    snapshots: list[dict[str, float]],
    start: str,
    end: str,
    initial_balance: float,
    log_file: str,
    config: StrategyRuntimeConfig | None = None,
) -> tuple[bool, dict[str, Any], int]:
    runtime_config = config or StrategyRuntimeConfig()
    try:
        validate_strategy_source(strategy_source)
    except StrategyValidationError as exc:
        return False, _build_error_payload(
            error_type=type(exc).__name__,
            message=str(exc),
            phase="strategy_validation",
            max_output_chars=runtime_config.max_captured_output_chars,
        ), 1

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_strategy_child_main,
        kwargs={
            "queue": queue,
            "strategy_source": strategy_source,
            "symbol": symbol,
            "raw_bars": raw_bars,
            "snapshots": snapshots,
            "start": start,
            "end": end,
            "initial_balance": initial_balance,
            "log_file": log_file,
            "max_output_chars": runtime_config.max_captured_output_chars,
        },
    )
    process.start()
    process.join(runtime_config.timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
        return False, _build_error_payload(
            error_type="StrategyTimeout",
            message=f"Strategy execution exceeded {runtime_config.timeout_seconds} seconds",
            phase="strategy_timeout",
            max_output_chars=runtime_config.max_captured_output_chars,
            context={"timeout_seconds": runtime_config.timeout_seconds},
        ), 1

    if not queue.empty():
        result = queue.get()
        if result.get("ok"):
            return True, result["summary"], 0
        return False, result["error"], 1

    return False, _build_error_payload(
        error_type="StrategyRuntimeError",
        message="Strategy subprocess exited without producing a result",
        phase="strategy_runtime",
        max_output_chars=runtime_config.max_captured_output_chars,
        context={"exit_code": process.exitcode},
    ), 1
