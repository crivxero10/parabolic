from dataclasses import dataclass, field
from enum import StrEnum
import logging
import math
from typing import Any

from parabolic.orchestrator import TradingContext
from parabolic.indicators import Indicators

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = True

Bar = dict[str, Any]


class Regime(StrEnum):
    BULL = "bull"
    BEAR = "bear"
    CRAB = "crab"


class Parity(StrEnum):
    GREEN = "green"
    RED = "red"
    DOJI = "doji"


class AssetSymbol(StrEnum):
    SPXL = "SPXL"
    SPXS = "SPXS"


class TradeAction(StrEnum):
    CLOSE_SHORT = "close_short"
    OPEN_LONG = "open_long"
    CLOSE_LONG = "close_long"
    OPEN_SHORT_PROXY = "open_short_proxy"
    CLOSE_LONG_ON_CRAB = "close_long_on_crab"
    CLOSE_SHORT_ON_CRAB = "close_short_on_crab"
    LONG_STOP_EXIT = "long_stop_exit"
    SHORT_STOP_EXIT = "short_stop_exit"


@dataclass(frozen=True)
class RegimeClassifierConfig:
    k_st: int = 6
    k_lt: int = 42
    lookback: int = 11
    crab_lower_bound: float = -2.0
    crab_upper_bound: float = 2.0
    quant_fold_factor: float = 1
    rolling_stop_pct: float = -0.1


@dataclass
class RegimeSessionState:
    session_date: str | None = None
    current_regime: str = Regime.CRAB
    entry_price_long: float | None = None
    entry_price_short: float | None = None
    momentum: int = 0
    current_parity: str = Parity.DOJI
    bars: list[Bar] = field(default_factory=list)
    blue_curve: list[float] = field(default_factory=list)
    green_curve: list[float] = field(default_factory=list)

    def reset(self, session_date: str) -> None:
        self.session_date = session_date
        self.current_regime = Regime.CRAB
        self.entry_price_long = None
        self.entry_price_short = None
        self.momentum = 0
        self.current_parity = Parity.DOJI
        self.bars.clear()
        self.blue_curve.clear()
        self.green_curve.clear()


class RegimeClassifier:
    def __init__(self, config: RegimeClassifierConfig | None = None):
        self.config = config or RegimeClassifierConfig()
        self.state = RegimeSessionState()

    def reset_session(self, session_date: str) -> None:
        self.state.reset(session_date)
        logger.info("session_reset session_date=%s", session_date)

    @staticmethod
    def _next_momentum_state(previous_bar: Bar | None, current_bar: Bar, current_momentum: int, current_parity: str) -> tuple[int, str]:
        if previous_bar is None:
            return current_momentum, current_parity

        bar_parity = Parity.GREEN if previous_bar["c"] < current_bar["c"] else Parity.RED
        if bar_parity == current_parity:
            return current_momentum + 1, current_parity
        return 0, bar_parity

    @classmethod
    def compute_next_state(cls, state: RegimeSessionState, bar: Bar, config: RegimeClassifierConfig) -> RegimeSessionState:
        previous_bar = state.bars[-1] if state.bars else None
        next_momentum, next_parity = cls._next_momentum_state(
            previous_bar=previous_bar,
            current_bar=bar,
            current_momentum=state.momentum,
            current_parity=state.current_parity,
        )

        bars = [*state.bars, bar]
        ongoing_series_close = [existing_bar["c"] for existing_bar in bars]
        ongoing_series_high = [existing_bar["h"] for existing_bar in bars]

        next_blue = (
            Indicators.ema_window(config.k_st, ongoing_series_high)
            if len(ongoing_series_high) >= config.k_st
            else ongoing_series_high[-1]
        )
        next_green = (
            Indicators.ema_window(config.k_lt, ongoing_series_close)
            if len(ongoing_series_close) >= config.k_lt
            else ongoing_series_close[-1]
        )

        next_state = RegimeSessionState(
            session_date=state.session_date,
            current_regime=state.current_regime,
            entry_price_long=state.entry_price_long,
            entry_price_short=state.entry_price_short,
            momentum=next_momentum,
            current_parity=next_parity,
            bars=bars,
            blue_curve=[*state.blue_curve, next_blue],
            green_curve=[*state.green_curve, next_green],
        )

        logger.info(
            "bar_ingested timestamp=%s close=%s high=%s momentum=%s parity=%s bars_in_session=%s",
            bar.get("t"),
            bar.get("c"),
            bar.get("h"),
            next_state.momentum,
            next_state.current_parity,
            len(next_state.bars),
        )
        logger.info(
            "curve_update timestamp=%s blue=%.6f green=%.6f warmup_threshold=%s warmed_up=%s",
            bar.get("t"),
            next_state.blue_curve[-1],
            next_state.green_curve[-1],
            config.k_lt,
            cls.is_warmed_up(next_state, config),
        )
        return next_state

    @staticmethod
    def is_warmed_up(state: RegimeSessionState, config: RegimeClassifierConfig) -> bool:
        minute_index = len(state.bars) - 1
        return minute_index > config.k_lt

    @staticmethod
    def get_regime(state: RegimeSessionState, config: RegimeClassifierConfig) -> str:
        diffs = [
            blue - green
            for blue, green in zip(
                state.blue_curve[-config.lookback:],
                state.green_curve[-config.lookback:],
            )
        ]

        anet = 0.0
        delta_x = 1.0
        for index in range(1, len(diffs)):
            anet += delta_x * (diffs[index] + diffs[index - 1]) / 2

        if config.crab_lower_bound <= anet <= config.crab_upper_bound:
            regime = Regime.CRAB
        elif anet > config.crab_upper_bound:
            regime = Regime.CRAB if state.momentum >= 3 and state.current_parity == Parity.RED else Regime.BULL
        elif state.momentum >= 3 and state.current_parity == Parity.GREEN:
            regime = Regime.CRAB
        else:
            regime = Regime.BEAR

        logger.info(
            "regime_evaluated lookback=%s diffs=%s area=%.6f crab_lb=%.6f crab_ub=%.6f momentum=%s parity=%s regime=%s",
            config.lookback,
            [round(diff, 6) for diff in diffs],
            anet,
            config.crab_lower_bound,
            config.crab_upper_bound,
            state.momentum,
            state.current_parity,
            regime,
        )
        return regime

    def _log_strategy_state(self, ctx: TradingContext, proposed_regime: str, current_price_xl: float, current_price_xs: float) -> None:
        logger.info(
            "strategy_state session_date=%s t=%s current_regime=%s proposed_regime=%s available_cash=%.6f spxl_price=%.6f spxs_price=%.6f spxl_position=%s spxs_position=%s entry_long=%s entry_short=%s",
            self.state.session_date,
            ctx.t,
            self.state.current_regime,
            proposed_regime,
            ctx.brokerage.available_cash,
            current_price_xl,
            current_price_xs,
            ctx.brokerage.positions.get(AssetSymbol.SPXL, 0),
            ctx.brokerage.positions.get(AssetSymbol.SPXS, 0),
            self.state.entry_price_long,
            self.state.entry_price_short,
        )

    @staticmethod
    def _execute_and_log(brokerage: Any, symbol: str, units: int, price: float, action: str) -> bool:
        did_execute = brokerage.execute(symbol, units, price)
        logger.info(
            "trade_decision action=%s symbol=%s units=%s price=%.6f executed=%s",
            action,
            symbol,
            units,
            price,
            did_execute,
        )
        return did_execute

    @staticmethod
    def _position_to_close(brokerage: Any, symbol: str) -> int:
        return -1 * brokerage.positions[symbol]

    def _defer_long_stop(self, brokerage: Any, units: int, entry_price: float) -> None:
        stop_price = entry_price * (1 + self.config.rolling_stop_pct)

        def activate(ctx: TradingContext) -> bool:
            current_price = ctx.market[ctx.t][AssetSymbol.SPXL]
            return current_price <= stop_price

        did_defer = brokerage.defer(
            asset_name=AssetSymbol.SPXL,
            units=-1 * units,
            target_price=stop_price,
            activate=activate,
        )
        logger.info(
            "defer_stop side=long symbol=%s units=%s entry=%.6f stop=%.6f deferred=%s",
            AssetSymbol.SPXL,
            units,
            entry_price,
            stop_price,
            did_defer,
        )

    def _defer_short_stop(self, brokerage: Any, units: int, entry_price: float) -> None:
        stop_price = entry_price * (1 + self.config.rolling_stop_pct)

        def activate(ctx: TradingContext) -> bool:
            current_price = ctx.market[ctx.t][AssetSymbol.SPXS]
            return current_price <= stop_price

        did_defer = brokerage.defer(
            asset_name=AssetSymbol.SPXS,
            units=-1 * units,
            target_price=stop_price,
            activate=activate,
        )
        logger.info(
            "defer_stop side=short_proxy symbol=%s units=%s entry=%.6f stop=%.6f deferred=%s",
            AssetSymbol.SPXS,
            units,
            entry_price,
            stop_price,
            did_defer,
        )

    def _transition_to_bull(self, brokerage: Any, current_price_xl: float, current_price_xs: float) -> None:
        if brokerage.positions.get(AssetSymbol.SPXS, 0) > 0:
            units_to_sell = self._position_to_close(brokerage, AssetSymbol.SPXS)
            self._execute_and_log(brokerage, AssetSymbol.SPXS, units_to_sell, current_price_xs, TradeAction.CLOSE_SHORT)
            self.state.entry_price_short = None

        trade_size = math.floor(brokerage.available_cash / current_price_xl)
        logger.info(
            "trade_sizing action=%s symbol=%s available_cash=%.6f price=%.6f trade_size=%s",
            TradeAction.OPEN_LONG,
            AssetSymbol.SPXL,
            brokerage.available_cash,
            current_price_xl,
            trade_size,
        )
        if trade_size > 0:
            if self._execute_and_log(brokerage, AssetSymbol.SPXL, trade_size, current_price_xl, TradeAction.OPEN_LONG):
                self.state.entry_price_long = current_price_xl
                self._defer_long_stop(brokerage, trade_size, current_price_xl)
        else:
            logger.info("trade_skip action=%s symbol=%s reason=trade_size_zero", TradeAction.OPEN_LONG, AssetSymbol.SPXL)

    def _transition_to_bear(self, brokerage: Any, current_price_xl: float, current_price_xs: float) -> None:
        if brokerage.positions.get(AssetSymbol.SPXL, 0) > 0:
            units_to_sell = self._position_to_close(brokerage, AssetSymbol.SPXL)
            self._execute_and_log(brokerage, AssetSymbol.SPXL, units_to_sell, current_price_xl, TradeAction.CLOSE_LONG)
            self.state.entry_price_long = None

        trade_size = math.floor(brokerage.available_cash / current_price_xs)
        logger.info(
            "trade_sizing action=%s symbol=%s available_cash=%.6f price=%.6f trade_size=%s",
            TradeAction.OPEN_SHORT_PROXY,
            AssetSymbol.SPXS,
            brokerage.available_cash,
            current_price_xs,
            trade_size,
        )
        if trade_size > 0:
            if self._execute_and_log(brokerage, AssetSymbol.SPXS, trade_size, current_price_xs, TradeAction.OPEN_SHORT_PROXY):
                self.state.entry_price_short = current_price_xs
                self._defer_short_stop(brokerage, trade_size, current_price_xs)
        else:
            logger.info("trade_skip action=%s symbol=%s reason=trade_size_zero", TradeAction.OPEN_SHORT_PROXY, AssetSymbol.SPXS)

    def _transition_to_crab(self, brokerage: Any, current_price_xl: float, current_price_xs: float) -> None:
        if brokerage.positions.get(AssetSymbol.SPXL, 0) > 0:
            units_to_sell = self._position_to_close(brokerage, AssetSymbol.SPXL)
            self._execute_and_log(brokerage, AssetSymbol.SPXL, units_to_sell, current_price_xl, TradeAction.CLOSE_LONG_ON_CRAB)
            self.state.entry_price_long = None
        if brokerage.positions.get(AssetSymbol.SPXS, 0) > 0:
            units_to_sell = self._position_to_close(brokerage, AssetSymbol.SPXS)
            self._execute_and_log(brokerage, AssetSymbol.SPXS, units_to_sell, current_price_xs, TradeAction.CLOSE_SHORT_ON_CRAB)
            self.state.entry_price_short = None

    def _handle_regime_change(self, brokerage: Any, regime: str, current_price_xl: float, current_price_xs: float, t: int) -> None:
        previous_regime = self.state.current_regime
        self.state.current_regime = regime
        logger.info(
            "regime_changed session_date=%s t=%s from_regime=%s to_regime=%s",
            self.state.session_date,
            t,
            previous_regime,
            regime,
        )

        if regime == Regime.BULL:
            self._transition_to_bull(brokerage, current_price_xl, current_price_xs)
        elif regime == Regime.BEAR:
            self._transition_to_bear(brokerage, current_price_xl, current_price_xs)
        else:
            self._transition_to_crab(brokerage, current_price_xl, current_price_xs)

    def apply_strategy(self, ctx: TradingContext) -> None:
        bar = getattr(ctx, "bar", None)
        if not bar:
            logger.info("strategy_skip reason=no_bar t=%s", getattr(ctx, "t", None))
            return

        session_date = str(bar.get("t", ""))[:10]
        if self.state.session_date != session_date:
            self.reset_session(session_date)

        self.state = self.compute_next_state(self.state, bar, self.config)
        if not self.is_warmed_up(self.state, self.config):
            logger.info(
                "strategy_skip reason=warmup_incomplete session_date=%s t=%s bars_in_session=%s required_gt=%s",
                session_date,
                ctx.t,
                len(self.state.bars),
                self.config.k_lt,
            )
            return

        brokerage = ctx.brokerage
        current_price_xl = ctx.market[ctx.t][AssetSymbol.SPXL]
        current_price_xs = ctx.market[ctx.t][AssetSymbol.SPXS]
        regime = self.get_regime(self.state, self.config)

        self._log_strategy_state(ctx, regime, current_price_xl, current_price_xs)

        if regime != self.state.current_regime:
            self._handle_regime_change(brokerage, regime, current_price_xl, current_price_xs, ctx.t)
            return

        logger.info(
            "regime_unchanged session_date=%s t=%s regime=%s stop_management=brokerage_deferred",
            session_date,
            ctx.t,
            regime,
        )
