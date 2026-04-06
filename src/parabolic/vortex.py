from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import StrEnum
import logging
import math
from typing import Any
from zoneinfo import ZoneInfo

from parabolic.classifier import Regime
from parabolic.mdp import MarketDataProvider
from parabolic.walker import Walker

logger = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")
MINUTES_PER_SESSION = 390


class VortexRegimeBias(StrEnum):
    SOFT = "soft"
    STRONG = "strong"


@dataclass(slots=True)
class VortexRequest:
    symbol: str
    regime: str | Regime
    start_date: str | None = None
    end_date: str | None = None
    length_days: int | None = None
    timeframe: str = "1Min"
    adjustment: str = "all"
    feed: str | None = None
    seed: int | float | str | bytes | bytearray | None = None
    regime_bias: str | VortexRegimeBias = VortexRegimeBias.SOFT
    reference_lookback_years: int = 5

    def __post_init__(self) -> None:
        self.symbol = self.symbol.strip().upper()
        if not self.symbol:
            raise ValueError("symbol is required")

        if self.timeframe != "1Min":
            raise ValueError("Vortex only supports regular-session 1Min generation")

        self.regime = Regime(str(self.regime).lower())
        self.regime_bias = VortexRegimeBias(str(self.regime_bias).lower())

        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("start_date and end_date must either both be set or both be omitted")

        if self.start_date is None and self.length_days is None:
            raise ValueError("length_days is required when start_date and end_date are omitted")

        if self.length_days is not None and self.length_days <= 0:
            raise ValueError("length_days must be positive")

        if self.reference_lookback_years <= 0:
            raise ValueError("reference_lookback_years must be positive")

    @property
    def is_dated(self) -> bool:
        return self.start_date is not None and self.end_date is not None


@dataclass(slots=True)
class Vortex:
    symbol: str
    bars: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.bars)


@dataclass(slots=True)
class _TemplateBar:
    body_return: float
    micro_gap_return: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    volume: float
    trade_count: float
    vw_position: float


@dataclass(slots=True)
class _SessionTemplate:
    bars: list[_TemplateBar]
    open_gap_return: float
    close_return: float
    realized_volatility: float
    regime_label: Regime
    close_price: float


@dataclass(slots=True)
class _TrainingSet:
    primary_sessions: list[list[dict[str, Any]]]
    reference_sessions: list[list[dict[str, Any]]]
    target_dates: list[str]
    start_price: float


class VortexGenerator:
    """Generate synthetic Alpaca-like intraday bars for a ticker.

    `regime_bias="soft"` keeps the ticker's learned historical character as the
    dominant force and tilts drift, volatility, and mode transitions toward the
    requested regime.

    `regime_bias="strong"` still preserves realistic intraday oscillation, but
    it applies materially stronger directional and volatility guidance so the
    realized path lands closer to the requested regime.
    """

    def __init__(self, market_data_provider: MarketDataProvider):
        self.market_data_provider = market_data_provider

    def generate(self, request: VortexRequest) -> Vortex:
        walker = Walker(seed=request.seed)
        training = self._build_training_set(request)
        templates = self._build_session_templates(
            primary_sessions=training.primary_sessions,
            reference_sessions=training.reference_sessions,
        )
        bars = self._generate_bars(
            request=request,
            walker=walker,
            session_templates=templates,
            target_dates=training.target_dates,
            start_price=training.start_price,
        )

        logger.info(
            "vortex_generated symbol=%s bars=%s sessions=%s regime=%s regime_bias=%s seed=%s primary_sessions=%s reference_sessions=%s",
            request.symbol,
            len(bars),
            len(training.target_dates),
            request.regime,
            request.regime_bias,
            walker.seed,
            len(training.primary_sessions),
            len(training.reference_sessions),
        )
        return Vortex(symbol=request.symbol, bars=bars)

    def _build_training_set(self, request: VortexRequest) -> _TrainingSet:
        if request.is_dated:
            requested_bars = self.market_data_provider.get_bars(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start=request.start_date,
                end=request.end_date,
                adjustment=request.adjustment,
                feed=request.feed,
            )
            target_dates = self._extract_session_dates(requested_bars)
            if not target_dates:
                raise ValueError(f"No market data found for {request.symbol} between {request.start_date} and {request.end_date}")

            primary_sessions = self._fetch_previous_sessions(
                symbol=request.symbol,
                before_date=target_dates[0],
                session_count=len(target_dates),
                adjustment=request.adjustment,
                feed=request.feed,
            )
            reference_sessions = self._fetch_reference_sessions(
                symbol=request.symbol,
                anchor_date=target_dates[0],
                lookback_years=request.reference_lookback_years,
                adjustment=request.adjustment,
                feed=request.feed,
            )
        else:
            target_dates = self._build_synthetic_dates(request.length_days or 0)
            anchor_date = self._resolve_reference_anchor_date(
                symbol=request.symbol,
                feed=request.feed,
            )
            reference_sessions = self._fetch_reference_sessions(
                symbol=request.symbol,
                anchor_date=anchor_date,
                lookback_years=request.reference_lookback_years,
                adjustment=request.adjustment,
                feed=request.feed,
            )
            if not reference_sessions:
                raise ValueError(f"No historical market data found for {request.symbol} in the last {request.reference_lookback_years} years")
            primary_sessions = reference_sessions[-min(len(reference_sessions), request.length_days or 0):]

        if not primary_sessions:
            raise ValueError(f"Unable to collect seed sessions for {request.symbol}")

        start_price = float(primary_sessions[-1][-1]["c"])
        logger.info(
            "vortex_training_ready symbol=%s target_sessions=%s primary_sessions=%s reference_sessions=%s start_price=%.6f",
            request.symbol,
            len(target_dates),
            len(primary_sessions),
            len(reference_sessions),
            start_price,
        )
        return _TrainingSet(
            primary_sessions=primary_sessions,
            reference_sessions=reference_sessions,
            target_dates=target_dates,
            start_price=start_price,
        )

    def _generate_bars(
        self,
        *,
        request: VortexRequest,
        walker: Walker,
        session_templates: list[_SessionTemplate],
        target_dates: list[str],
        start_price: float,
    ) -> list[dict[str, Any]]:
        if not session_templates:
            raise ValueError("Vortex generation requires at least one session template")

        bars: list[dict[str, Any]] = []
        previous_close = max(start_price, 0.01)
        bias_strength = 0.35 if request.regime_bias == VortexRegimeBias.SOFT else 0.75
        target_pool = [template for template in session_templates if template.regime_label == request.regime]
        if not target_pool:
            target_pool = session_templates

        for session_date in target_dates:
            base_template = walker.weighted_choice(
                items=[target_pool, session_templates],
                weights=[0.7 if request.regime_bias == VortexRegimeBias.STRONG else 0.55, 0.3 if request.regime_bias == VortexRegimeBias.STRONG else 0.45],
            )
            template = walker.choice(base_template)
            target_close_return = self._sample_target_close_return(
                walker=walker,
                regime=request.regime,
                regime_bias=request.regime_bias,
                templates=target_pool,
            )
            volatility_scale = self._sample_volatility_scale(
                walker=walker,
                regime=request.regime,
                regime_bias=request.regime_bias,
                template=template,
            )
            open_gap = self._blend_value(
                base_value=template.open_gap_return,
                target_value=self._target_open_gap(request.regime),
                strength=bias_strength * 0.5,
            )
            session_open = max(previous_close * math.exp(open_gap), 0.01)
            previous_close = self._generate_session(
                bars=bars,
                walker=walker,
                session_date=session_date,
                previous_close=previous_close,
                session_open=session_open,
                session_template=template,
                target_close_return=target_close_return,
                volatility_scale=volatility_scale,
                regime=request.regime,
                bias_strength=bias_strength,
            )

        return bars

    def _generate_session(
        self,
        *,
        bars: list[dict[str, Any]],
        walker: Walker,
        session_date: str,
        previous_close: float,
        session_open: float,
        session_template: _SessionTemplate,
        target_close_return: float,
        volatility_scale: float,
        regime: Regime,
        bias_strength: float,
    ) -> float:
        current_close = session_open
        realized_log_return = 0.0
        mode = self._initial_mode_for_regime(regime)
        session_anchor = session_open

        for minute_index, template_bar in enumerate(session_template.bars):
            remaining_minutes = MINUTES_PER_SESSION - minute_index
            mode = self._transition_mode(
                walker=walker,
                current_mode=mode,
                regime=regime,
                bias_strength=bias_strength,
            )
            micro_gap = template_bar.micro_gap_return * (0.5 + 0.5 * volatility_scale)
            if minute_index == 0:
                bar_open = session_open
            else:
                bar_open = max(current_close * math.exp(micro_gap), 0.01)

            guidance = (target_close_return - realized_log_return) / max(remaining_minutes, 1)
            mode_drift = self._mode_drift(mode, regime) * bias_strength
            mean_reversion = self._mean_reversion_pressure(
                regime=regime,
                bar_open=bar_open,
                session_anchor=session_anchor,
            )
            noise = walker.gauss(0.0, max(abs(template_bar.body_return), 0.00035) * 0.35)
            body_return = (
                (template_bar.body_return * volatility_scale)
                + mode_drift
                + (guidance * (0.5 + bias_strength))
                - mean_reversion
                + noise
            )
            body_return = max(min(body_return, 0.08), -0.08)
            bar_close = max(bar_open * math.exp(body_return), 0.01)

            wick_scale = max(volatility_scale, 0.35)
            upper_wick = max(template_bar.upper_wick_ratio * wick_scale, 0.00005)
            lower_wick = max(template_bar.lower_wick_ratio * wick_scale, 0.00005)
            bar_high = max(bar_open, bar_close) * (1.0 + upper_wick)
            bar_low = min(bar_open, bar_close) * max(1.0 - lower_wick, 0.01)

            volume_scale = 1.0 + min(abs(body_return) * 24.0, 2.5)
            volume = max(int(round(template_bar.volume * volume_scale * (0.75 + walker.random() * 0.5))), 1)
            trades = max(int(round(template_bar.trade_count * volume_scale * (0.80 + walker.random() * 0.4))), 1)
            midpoint = (bar_open + bar_close) / 2.0
            price_span = max(bar_high - bar_low, 0.0001)
            vw = midpoint + (template_bar.vw_position * price_span * 0.5)
            vw = min(max(vw, bar_low), bar_high)

            bars.append(
                {
                    "t": self._minute_timestamp_utc(session_date, minute_index),
                    "o": round(bar_open, 4),
                    "h": round(bar_high, 4),
                    "l": round(bar_low, 4),
                    "c": round(bar_close, 4),
                    "v": volume,
                    "vw": round(vw, 4),
                    "n": trades,
                }
            )
            current_close = bar_close
            realized_log_return += math.log(max(bar_close, 0.01) / max(bar_open, 0.01))

        return current_close

    def _build_session_templates(
        self,
        *,
        primary_sessions: list[list[dict[str, Any]]],
        reference_sessions: list[list[dict[str, Any]]],
    ) -> list[_SessionTemplate]:
        sessions = [*reference_sessions, *primary_sessions]
        templates: list[_SessionTemplate] = []
        for session in sessions:
            template = self._session_to_template(session)
            if template is not None:
                templates.append(template)
        return templates

    def _session_to_template(self, session: list[dict[str, Any]]) -> _SessionTemplate | None:
        if len(session) != MINUTES_PER_SESSION:
            return None

        template_bars: list[_TemplateBar] = []
        intraday_returns: list[float] = []
        prior_close = float(session[0]["o"])
        for raw_bar in session:
            open_price = max(float(raw_bar.get("o", prior_close)), 0.01)
            high_price = max(float(raw_bar.get("h", open_price)), open_price)
            low_price = min(float(raw_bar.get("l", open_price)), open_price)
            close_price = max(float(raw_bar.get("c", open_price)), 0.01)

            body_return = math.log(close_price / open_price)
            micro_gap_return = math.log(open_price / max(prior_close, 0.01))
            upper_wick_ratio = max(high_price - max(open_price, close_price), 0.0) / max(max(open_price, close_price), 0.01)
            lower_wick_ratio = max(min(open_price, close_price) - low_price, 0.0) / max(min(open_price, close_price), 0.01)
            volume = max(float(raw_bar.get("v", 1.0)), 1.0)
            trade_count = max(float(raw_bar.get("n", 1.0)), 1.0)
            span = max(high_price - low_price, 0.0001)
            midpoint = (open_price + close_price) / 2.0
            vw = float(raw_bar.get("vw", midpoint))
            vw_position = max(min((vw - midpoint) / span, 1.0), -1.0)

            template_bars.append(
                _TemplateBar(
                    body_return=body_return,
                    micro_gap_return=micro_gap_return,
                    upper_wick_ratio=upper_wick_ratio,
                    lower_wick_ratio=lower_wick_ratio,
                    volume=volume,
                    trade_count=trade_count,
                    vw_position=vw_position,
                )
            )
            intraday_returns.append(body_return)
            prior_close = close_price

        session_open = max(float(session[0]["o"]), 0.01)
        previous_close = max(float(session[0]["o"]) / math.exp(template_bars[0].micro_gap_return), 0.01)
        session_close = max(float(session[-1]["c"]), 0.01)
        close_return = math.log(session_close / session_open)
        open_gap_return = math.log(session_open / previous_close)
        realized_volatility = self._stdev(intraday_returns)

        return _SessionTemplate(
            bars=template_bars,
            open_gap_return=open_gap_return,
            close_return=close_return,
            realized_volatility=realized_volatility,
            regime_label=self._classify_session_regime(close_return, realized_volatility),
            close_price=session_close,
        )

    def _fetch_previous_sessions(
        self,
        *,
        symbol: str,
        before_date: str,
        session_count: int,
        adjustment: str,
        feed: str | None,
    ) -> list[list[dict[str, Any]]]:
        if session_count <= 0:
            return []

        end_day = date.fromisoformat(before_date) - timedelta(days=1)
        start_day = end_day - timedelta(days=max(session_count * 4, 30))
        sessions: list[list[dict[str, Any]]] = []

        while len(sessions) < session_count:
            bars = self.market_data_provider.get_bars(
                symbol=symbol,
                timeframe="1Min",
                start=start_day.isoformat(),
                end=end_day.isoformat(),
                adjustment=adjustment,
                feed=feed,
            )
            sessions = self._split_sessions(bars)
            if len(sessions) >= session_count or start_day.year <= 1995:
                break
            start_day -= timedelta(days=max(session_count * 2, 30))

        return sessions[-session_count:]

    def _fetch_reference_sessions(
        self,
        *,
        symbol: str,
        anchor_date: str,
        lookback_years: int,
        adjustment: str,
        feed: str | None,
    ) -> list[list[dict[str, Any]]]:
        anchor = date.fromisoformat(anchor_date)
        start = anchor - timedelta(days=lookback_years * 366)
        end = anchor - timedelta(days=1)
        if end < start:
            return []
        bars = self.market_data_provider.get_bars(
            symbol=symbol,
            timeframe="1Min",
            start=start.isoformat(),
            end=end.isoformat(),
            adjustment=adjustment,
            feed=feed,
        )
        return self._split_sessions(bars)

    def _resolve_reference_anchor_date(self, *, symbol: str, feed: str | None) -> str:
        latest_bar = self.market_data_provider.get_latest_bar(symbol=symbol, feed=feed)
        if latest_bar and latest_bar.get("t"):
            return self._extract_trading_date(str(latest_bar["t"]))
        return date.today().isoformat()

    def _extract_session_dates(self, bars: list[dict[str, Any]]) -> list[str]:
        return [self._extract_trading_date(session[0]["t"]) for session in self._split_sessions(bars)]

    def _split_sessions(self, bars: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        sessions_by_date: dict[str, list[dict[str, Any]]] = {}
        ordered_dates: list[str] = []
        for bar in bars:
            timestamp = bar.get("t")
            if timestamp is None:
                continue
            session_date = self._extract_trading_date(str(timestamp))
            if session_date not in sessions_by_date:
                sessions_by_date[session_date] = []
                ordered_dates.append(session_date)
            sessions_by_date[session_date].append(bar)
        sessions = [sessions_by_date[session_date] for session_date in ordered_dates]
        return [session for session in sessions if len(session) == MINUTES_PER_SESSION]

    def _build_synthetic_dates(self, length_days: int) -> list[str]:
        synthetic_dates: list[str] = []
        current_day = date(3000, 1, 3)
        while len(synthetic_dates) < length_days:
            if current_day.weekday() < 5:
                synthetic_dates.append(current_day.isoformat())
            current_day += timedelta(days=1)
        return synthetic_dates

    def _sample_target_close_return(
        self,
        *,
        walker: Walker,
        regime: Regime,
        regime_bias: VortexRegimeBias,
        templates: list[_SessionTemplate],
    ) -> float:
        close_returns = [template.close_return for template in templates]
        if not close_returns:
            close_returns = [0.0]
        empirical = walker.choice(close_returns)
        bias_map = {
            Regime.BULL: 0.0022,
            Regime.BEAR: -0.0025,
            Regime.CRAB: 0.0,
        }
        strength = 0.35 if regime_bias == VortexRegimeBias.SOFT else 0.75
        return self._blend_value(empirical, bias_map[regime], strength)

    def _sample_volatility_scale(
        self,
        *,
        walker: Walker,
        regime: Regime,
        regime_bias: VortexRegimeBias,
        template: _SessionTemplate,
    ) -> float:
        base = 1.0 + walker.uniform(-0.12, 0.18)
        if regime == Regime.BEAR:
            base += 0.20 if regime_bias == VortexRegimeBias.STRONG else 0.10
        elif regime == Regime.CRAB:
            base -= 0.08 if regime_bias == VortexRegimeBias.STRONG else 0.03
        if template.realized_volatility > 0.002:
            base += min(template.realized_volatility * 20.0, 0.25)
        return max(base, 0.35)

    def _target_open_gap(self, regime: Regime) -> float:
        return {
            Regime.BULL: 0.0003,
            Regime.BEAR: -0.00035,
            Regime.CRAB: 0.0,
        }[regime]

    def _classify_session_regime(self, close_return: float, realized_volatility: float) -> Regime:
        score = close_return / max(realized_volatility, 0.0015)
        if score > 0.45:
            return Regime.BULL
        if score < -0.45:
            return Regime.BEAR
        return Regime.CRAB

    def _initial_mode_for_regime(self, regime: Regime) -> str:
        return {
            Regime.BULL: "trend_up",
            Regime.BEAR: "trend_down",
            Regime.CRAB: "mean_revert",
        }[regime]

    def _transition_mode(
        self,
        *,
        walker: Walker,
        current_mode: str,
        regime: Regime,
        bias_strength: float,
    ) -> str:
        modes = ["trend_up", "trend_down", "mean_revert", "volatile", "pause"]
        base_weights: dict[Regime, dict[str, float]] = {
            Regime.BULL: {
                "trend_up": 0.36 + bias_strength * 0.22,
                "trend_down": 0.10,
                "mean_revert": 0.23,
                "volatile": 0.17,
                "pause": 0.14,
            },
            Regime.BEAR: {
                "trend_up": 0.11,
                "trend_down": 0.36 + bias_strength * 0.22,
                "mean_revert": 0.19,
                "volatile": 0.22,
                "pause": 0.12,
            },
            Regime.CRAB: {
                "trend_up": 0.17,
                "trend_down": 0.17,
                "mean_revert": 0.34 + bias_strength * 0.18,
                "volatile": 0.16,
                "pause": 0.16,
            },
        }
        weights = base_weights[regime]
        weights[current_mode] += 0.18
        return str(walker.weighted_choice(modes, [weights[mode] for mode in modes]))

    def _mode_drift(self, mode: str, regime: Regime) -> float:
        drift = {
            "trend_up": 0.00045,
            "trend_down": -0.00045,
            "mean_revert": 0.0,
            "volatile": 0.0,
            "pause": 0.0,
        }[mode]
        if mode == "volatile":
            return 0.00015 if regime == Regime.BULL else (-0.00015 if regime == Regime.BEAR else 0.0)
        return drift

    def _mean_reversion_pressure(self, *, regime: Regime, bar_open: float, session_anchor: float) -> float:
        if session_anchor <= 0:
            return 0.0
        deviation = math.log(max(bar_open, 0.01) / max(session_anchor, 0.01))
        if regime == Regime.CRAB:
            return deviation * 0.12
        if regime == Regime.BULL:
            return max(deviation, 0.0) * 0.04
        return min(deviation, 0.0) * 0.04

    def _blend_value(self, base_value: float, target_value: float, strength: float) -> float:
        clamped_strength = min(max(strength, 0.0), 1.0)
        return (base_value * (1.0 - clamped_strength)) + (target_value * clamped_strength)

    def _minute_timestamp_utc(self, session_date: str, minute_index: int) -> str:
        minute_dt = datetime.combine(
            date.fromisoformat(session_date),
            time(9, 30),
            tzinfo=MARKET_TZ,
        ) + timedelta(minutes=minute_index)
        return minute_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _extract_trading_date(self, timestamp: str) -> str:
        if "T" in timestamp:
            return timestamp.split("T", 1)[0]
        return timestamp[:10]

    def _stdev(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(max(variance, 0.0))
