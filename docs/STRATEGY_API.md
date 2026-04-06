# Strategy API

This document describes the contract for agent-authored and human-authored strategies used by the `evaluate` CLI workflow.

## Function Contract

Stdin strategies must define exactly one top-level function:

```python
def strategy(ctx):
    ...
```

No imports, `while` loops, `try/except`, `with` blocks, or dangerous builtin calls are allowed in stdin strategy mode.

## Execution Model

In stdin strategy mode, the strategy is intended for intraday evaluation on minute bars.

- The CLI loads the requested market data first.
- The strategy is then called once per visible bar.
- In the normal intraday workflow, that means one call per minute bar.
- `ctx.market` and `ctx.bars` contain only the data visible up to the current minute.
- `ctx.session_market` contains the full loaded session and should not be used for signal generation because it can introduce lookahead bias.

## Context Contract

Inside `strategy(ctx)`, the context exposes:

- `ctx.t`
  Current bar index within the loaded session window.
- `ctx.asset_name`
  Primary symbol from `--symbol`.
- `ctx.market`
  List of market snapshots up to and including the current bar. Each snapshot is a dict keyed by loaded symbol.
- `ctx.bar`
  Current raw bar dict for the primary symbol.
- `ctx.bars`
  Raw primary-symbol bars up to and including the current bar.
- `ctx.start_date`
- `ctx.end_date`
- `ctx.timeframe`
- `ctx.adjustment`
- `ctx.feed`
- `ctx.session_market`
  Entire loaded session snapshot sequence.
- `ctx.session_length`
  Number of snapshots in the session.
- `ctx.is_session_start`
- `ctx.is_session_end`
- `ctx.brokerage`
  Brokerage object used for order placement and portfolio inspection.

## Market Universe

By default, the strategy market universe is:

- primary symbol from `--symbol`
- `SPXL`
- `SPXS`

To override that explicitly, use:

```bash
--market-symbols SPY,SPXL,SPXS,VIXY
```

If `--market-symbols` is provided, every snapshot in `ctx.market` will include exactly the declared aligned universe, subject to alignment on timestamps.

## Brokerage API

Common strategy-facing brokerage members:

- `ctx.brokerage.balance`
- `ctx.brokerage.available_cash`
- `ctx.brokerage.positions`
- `ctx.brokerage.operations`
- `ctx.brokerage.deferred_instructions`

Common strategy-facing brokerage methods:

- `ctx.brokerage.execute(asset_name, units, price, timestamp=None)`
- `ctx.brokerage.defer(asset_name, units, target_price, activate, position_id=None)`
- `ctx.brokerage.liquidate(market_snapshot)`
- `ctx.brokerage.get_total_pnl(market_snapshot)`
- `ctx.brokerage.get_avg_cost_basis(asset_name)`

## Indicators API

Technical indicators are available from:

```python
Indicators
```

`Indicators` is injected into the stdin strategy runtime automatically. Do not import it.

Available methods:

- `Indicators.sma(series, n)`
- `Indicators.ema(series, alpha)`
- `Indicators.ema_window(n, series)`
- `Indicators.ema_area_between_curves(series, k_st, k_lt, lookback)`
- `Indicators.rolling_std(series, n)`
- `Indicators.bollinger_bands(series, n, num_std=2.0)`
- `Indicators.rsi(series, n=14)`
- `Indicators.true_range(highs, lows, closes)`
- `Indicators.atr(highs, lows, closes, n=14)`
- `Indicators.vwap(highs, lows, closes, volumes)`
- `Indicators.macd(series, fast=12, slow=26, signal=9)`
- `Indicators.stochastic_oscillator(highs, lows, closes, k_period=14, d_period=3)`
- `Indicators.williams_r(highs, lows, closes, n=14)`
- `Indicators.cci(highs, lows, closes, n=20)`
- `Indicators.mfi(highs, lows, closes, volumes, n=14)`
- `Indicators.obv(closes, volumes)`

## Example

```python
def strategy(ctx):
    closes = [float(row[ctx.asset_name]) for row in ctx.market]
    if len(closes) < 20:
        return

    sma20 = Indicators.sma(closes, 20)[-1]
    current_price = closes[-1]
    current_units = int(ctx.brokerage.positions.get(ctx.asset_name, 0))
    timestamp = ctx.bar["t"] if ctx.bar is not None else None

    if sma20 is not None and current_price > sma20 and current_units == 0:
        units = int(ctx.brokerage.available_cash // current_price)
        if units > 0:
            ctx.brokerage.execute(ctx.asset_name, units, current_price, timestamp=timestamp)

    if ctx.is_session_end and current_units > 0:
        ctx.brokerage.execute(ctx.asset_name, -current_units, current_price, timestamp=timestamp)
```

## Machine-Readable Spec

For agents, the same contract is available as JSON:

```bash
python -m parabolic.driver strategy-spec
```
