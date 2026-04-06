from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
import logging
import multiprocessing
import os
import random
from typing import Any, Callable, Literal, Sequence

from parabolic.backtest import Backtester
from parabolic.brokerage import Brokerage
from parabolic.orchestrator import ContextOrchestrator
from parabolic.risk import returns_from_equity_curve, summarize_risk_metrics

logger = logging.getLogger(__name__)

Strategy = Callable[[Any], None]
StrategyFactory = Callable[[dict[str, Any]], Strategy]
BrokerageFactory = Callable[[], Brokerage]


@dataclass(frozen=True)
class ParameterRange:
    name: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class TuningResult:
    strategy_name: str
    parameters: dict[str, Any]
    balances: list[float]
    labels: list[str]
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    final_balance: float


@dataclass(frozen=True)
class AdaptiveSearchConfig:
    initial_samples: int = 512
    refinement_rounds: int = 8
    top_k: int = 24
    exploratory_samples_per_round: int = 128
    max_full_search_combinations: int = 250_000
    per_parameter_coverage_multiplier: int = 16
    pairwise_anchor_samples_per_value: int = 2
    executor_kind: Literal["sequential", "thread", "process"] = "thread"
    max_parallelism: int | None = 64
    max_parallel_workers: int = 64
    parallel_batch_size: int = 32
    expand_numeric_parameters: bool = True
    integer_range_min: int = 1
    integer_range_max: int = 300
    float_negative_min: float = -1.0
    float_negative_max: float = -0.01
    float_mixed_sign_min: float = -10.0
    float_mixed_sign_max: float = 10.0
    float_positive_min: float = 0.0
    float_positive_max: float = 10.0
    numeric_expansion_steps_per_side: int = 2
    numeric_midpoints_per_gap: int = 1
    coarse_integer_points: int = 17
    coarse_float_points: int = 17
    neighbor_window: int = 2
    round_random_multiplier: int = 4
    random_seed: int = 42


class Tuner:
    def __init__(
        self,
        *,
        asset_name: str,
        context_orchestrator: ContextOrchestrator,
        brokerage_factory: BrokerageFactory,
        initial_balance: float,
        search_config: AdaptiveSearchConfig | None = None,
    ) -> None:
        self.asset_name = asset_name
        self.context_orchestrator = context_orchestrator
        self.brokerage_factory = brokerage_factory
        self.initial_balance = float(initial_balance)
        self.search_config = search_config or AdaptiveSearchConfig()

    @staticmethod
    def _parameter_key(
        parameter_space: Sequence[ParameterRange],
        parameters: dict[str, Any],
    ) -> tuple[Any, ...]:
        return tuple(parameters[parameter.name] for parameter in parameter_space)

    @staticmethod
    def _parameter_dict(
        parameter_space: Sequence[ParameterRange],
        values: Sequence[Any],
    ) -> dict[str, Any]:
        return {parameter.name: value for parameter, value in zip(parameter_space, values)}

    @staticmethod
    def _is_valid_parameter_combination(parameters: dict[str, Any]) -> bool:
        k_st = parameters.get("k_st")
        k_lt = parameters.get("k_lt")
        lookback = parameters.get("lookback")
        crab_lower_bound = parameters.get("crab_lower_bound")
        crab_upper_bound = parameters.get("crab_upper_bound")

        if k_st is None or k_lt is None:
            return True

        if not (k_st >= 3):
            return False
        if not (k_st < k_lt <= 200):
            return False

        if lookback is not None:
            max_lookback = int(float(k_lt) ** 0.5)
            if not (k_st <= lookback <= max_lookback):
                return False

        if crab_lower_bound is None or crab_upper_bound is None:
            return True

        crab_limit = float(k_st) ** 0.5
        if not (-crab_limit <= float(crab_lower_bound) <= crab_limit):
            return False
        if not (-crab_limit <= float(crab_upper_bound) <= crab_limit):
            return False
        return True

    @staticmethod
    def _total_combinations(parameter_space: Sequence[ParameterRange]) -> int:
        total = 1
        for parameter in parameter_space:
            total *= len(parameter.values)
        return total

    def _effective_parameter_space(
        self,
        parameter_space: Sequence[ParameterRange],
    ) -> list[ParameterRange]:
        allowed_names = {
            "k_st",
            "k_lt",
            "lookback",
            "crab_lower_bound",
            "crab_upper_bound",
        }
        effective_parameter_space: list[ParameterRange] = []
        for parameter in parameter_space:
            if parameter.name in allowed_names:
                effective_parameter_space.append(parameter)
            else:
                effective_parameter_space.append(
                    ParameterRange(name=parameter.name, values=(parameter.values[0],))
                )
        return effective_parameter_space

    @staticmethod
    def _score_result(result: TuningResult) -> tuple[float, float, float, float]:
        return (result.sharpe, result.sortino, result.calmar, result.final_balance)

    @staticmethod
    def _format_parameters(parameters: dict[str, Any]) -> str:
        return ", ".join(f"{name}={parameters[name]!r}" for name in sorted(parameters))

    @staticmethod
    def _coerce_numeric_value(template: Any, value: float) -> Any:
        if isinstance(template, bool):
            return template
        if isinstance(template, int) and not isinstance(template, bool):
            return int(round(value))
        return float(value)

    def _expand_numeric_values(
        self,
        values: tuple[Any, ...],
        config: AdaptiveSearchConfig,
    ) -> tuple[Any, ...]:
        if not values or not config.expand_numeric_parameters:
            return values
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            return values

        template = values[0]
        unique_sorted = sorted({float(value) for value in values})
        if len(unique_sorted) <= 1:
            if isinstance(template, int) and not isinstance(template, bool):
                lower_bound = config.integer_range_min
                upper_bound = config.integer_range_max
            elif all(value <= 0 for value in unique_sorted):
                lower_bound = config.float_negative_min
                upper_bound = config.float_negative_max
            elif all(value >= 0 for value in unique_sorted):
                lower_bound = config.float_positive_min
                upper_bound = config.float_positive_max
            else:
                lower_bound = config.float_mixed_sign_min
                upper_bound = config.float_mixed_sign_max
        else:
            if isinstance(template, int) and not isinstance(template, bool):
                lower_bound = config.integer_range_min
                upper_bound = config.integer_range_max
            elif all(value <= 0 for value in unique_sorted):
                lower_bound = config.float_negative_min
                upper_bound = config.float_negative_max
            elif all(value >= 0 for value in unique_sorted):
                lower_bound = config.float_positive_min
                upper_bound = config.float_positive_max
            else:
                lower_bound = config.float_mixed_sign_min
                upper_bound = config.float_mixed_sign_max

        if isinstance(template, int) and not isinstance(template, bool):
            point_count = max(config.coarse_integer_points, len(values))
            lower_bound = max(int(round(lower_bound)), 1)
            upper_bound = max(int(round(upper_bound)), lower_bound)
            if point_count <= 1:
                return (lower_bound,)
            step = (upper_bound - lower_bound) / (point_count - 1)
            expanded_ints = {
                max(int(round(lower_bound + step * index)), 1)
                for index in range(point_count)
            }
            expanded_ints.add(lower_bound)
            expanded_ints.add(upper_bound)
            return tuple(sorted(expanded_ints))

        point_count = max(config.coarse_float_points, len(values))
        if point_count <= 1:
            return (float(lower_bound),)
        step = (upper_bound - lower_bound) / (point_count - 1)
        expanded = {
            round(lower_bound + step * index, 10)
            for index in range(point_count)
        }
        expanded.add(round(lower_bound, 10))
        expanded.add(round(upper_bound, 10))
        return tuple(sorted(expanded))

    def _expand_parameter_space(
        self,
        parameter_space: Sequence[ParameterRange],
        config: AdaptiveSearchConfig,
    ) -> list[ParameterRange]:
        expanded_parameter_space: list[ParameterRange] = []
        for parameter in parameter_space:
            expanded_values = self._expand_numeric_values(parameter.values, config)

            if parameter.name == "k_st":
                expanded_values = tuple(
                    value for value in expanded_values
                    if isinstance(value, int) and value >= 3
                )
            elif parameter.name == "k_lt":
                expanded_values = tuple(
                    value for value in expanded_values
                    if isinstance(value, int) and value <= 200
                )
            elif parameter.name == "lookback":
                expanded_values = tuple(
                    value for value in expanded_values
                    if isinstance(value, int) and value >= 1
                )

            if not expanded_values:
                expanded_values = (parameter.values[0],)

            expanded_parameter_space.append(
                ParameterRange(name=parameter.name, values=expanded_values)
            )
        return expanded_parameter_space

    def _coverage_sample_count(
        self,
        parameter_space: Sequence[ParameterRange],
        config: AdaptiveSearchConfig,
    ) -> int:
        if not parameter_space:
            return 0
        total_values = sum(len(parameter.values) for parameter in parameter_space)
        return max(
            config.initial_samples,
            total_values * config.per_parameter_coverage_multiplier,
        )

    def _append_candidate_if_new(
        self,
        *,
        parameter_space: Sequence[ParameterRange],
        candidate: dict[str, Any],
        seen: set[tuple[Any, ...]],
        candidates: list[dict[str, Any]],
    ) -> None:
        key = self._parameter_key(parameter_space, candidate)
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    def _dedupe_parameter_sets(
        self,
        parameter_space: Sequence[ParameterRange],
        parameter_sets: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for parameters in parameter_sets:
            key = self._parameter_key(parameter_space, parameters)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(parameters)
        return deduped

    def _resolved_max_parallel_workers(self) -> int:
        configured_workers = (
            self.search_config.max_parallelism
            if self.search_config.max_parallelism is not None
            else self.search_config.max_parallel_workers
        )
        cpu_limit = max(1, min(32, (os.cpu_count() or 4)))
        return max(1, min(int(configured_workers), cpu_limit))

    @staticmethod
    def _chunk_parameter_sets(
        parameter_sets: Sequence[dict[str, Any]],
        chunk_size: int,
    ) -> list[list[dict[str, Any]]]:
        if chunk_size <= 0:
            return [list(parameter_sets)]
        return [
            list(parameter_sets[index : index + chunk_size])
            for index in range(0, len(parameter_sets), chunk_size)
        ]

    def _evaluate_parameter_batch_sequential(
        self,
        *,
        strategy_name: str,
        parameter_space: Sequence[ParameterRange],
        parameter_sets: Sequence[dict[str, Any]],
        strategy_factory: StrategyFactory,
    ) -> dict[tuple[Any, ...], TuningResult]:
        results_by_key: dict[tuple[Any, ...], TuningResult] = {}
        for parameters in parameter_sets:
            result = self.evaluate(
                strategy_name=strategy_name,
                parameters=parameters,
                strategy_factory=strategy_factory,
            )
            if result is None:
                continue
            key = self._parameter_key(parameter_space, parameters)
            results_by_key[key] = result
        return results_by_key

    def _evaluate_parameter_batch_parallel(
        self,
        *,
        strategy_name: str,
        parameter_space: Sequence[ParameterRange],
        parameter_sets: Sequence[dict[str, Any]],
        strategy_factory: StrategyFactory,
        round_label: str,
    ) -> dict[tuple[Any, ...], TuningResult]:
        deduped_parameter_sets = self._dedupe_parameter_sets(parameter_space, parameter_sets)
        if not deduped_parameter_sets:
            return {}

        max_workers = min(self._resolved_max_parallel_workers(), len(deduped_parameter_sets))
        configured_executor = self.search_config.executor_kind
        executor_kind = configured_executor
        if configured_executor == "process":
            logger.warning(
                "adaptive_search_process_executor_memory_risk strategy=%s round=%s workers=%s batch_size=%s; falling_back_to_threads",
                strategy_name,
                round_label,
                max_workers,
                self.search_config.parallel_batch_size,
            )
            executor_kind = "thread"

        logger.info(
            "adaptive_search_parallel_batch strategy=%s round=%s candidates=%s workers=%s executor=%s configured_parallelism=%s configured_workers=%s batch_size=%s",
            strategy_name,
            round_label,
            len(deduped_parameter_sets),
            max_workers,
            executor_kind,
            self.search_config.max_parallelism,
            self.search_config.max_parallel_workers,
            self.search_config.parallel_batch_size,
        )

        if executor_kind == "sequential" or max_workers <= 1:
            return self._evaluate_parameter_batch_sequential(
                strategy_name=strategy_name,
                parameter_space=parameter_space,
                parameter_sets=deduped_parameter_sets,
                strategy_factory=strategy_factory,
            )

        results_by_key: dict[tuple[Any, ...], TuningResult] = {}
        parameter_batches = self._chunk_parameter_sets(
            deduped_parameter_sets,
            self.search_config.parallel_batch_size,
        )

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for batch_index, parameter_batch in enumerate(parameter_batches, start=1):
                    logger.debug(
                        "adaptive_search_parallel_submitted strategy=%s round=%s batch=%s batch_candidates=%s",
                        strategy_name,
                        round_label,
                        batch_index,
                        len(parameter_batch),
                    )
                    futures = {
                        executor.submit(
                            self.evaluate,
                            strategy_name=strategy_name,
                            parameters=parameters,
                            strategy_factory=strategy_factory,
                        ): parameters
                        for parameters in parameter_batch
                    }
                    for future in as_completed(futures):
                        parameters = futures[future]
                        result = future.result()
                        if result is None:
                            continue
                        key = self._parameter_key(parameter_space, parameters)
                        results_by_key[key] = result
            return results_by_key
        except Exception as exc:
            logger.warning(
                "adaptive_search_parallel_fallback strategy=%s round=%s reason=%s",
                strategy_name,
                round_label,
                exc,
            )
            return self._evaluate_parameter_batch_sequential(
                strategy_name=strategy_name,
                parameter_space=parameter_space,
                parameter_sets=deduped_parameter_sets,
                strategy_factory=strategy_factory,
            )

    def _build_initial_candidates(
        self,
        parameter_space: Sequence[ParameterRange],
        config: AdaptiveSearchConfig,
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        total = self._total_combinations(parameter_space)
        raw_target = min(total, self._coverage_sample_count(parameter_space, config))
        target = max(1, raw_target)
        if total <= config.max_full_search_combinations:
            logger.info(
                "adaptive_search_mode mode=full_grid total_combinations=%s",
                total,
            )
            return self._sample_parameter_sets(parameter_space, total, rng, set())

        candidates: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()

        anchor_parameters = {
            parameter.name: parameter.values[len(parameter.values) // 2]
            for parameter in parameter_space
        }
        if self._is_valid_parameter_combination(anchor_parameters):
            self._append_candidate_if_new(
                parameter_space=parameter_space,
                candidate=dict(anchor_parameters),
                seen=seen,
                candidates=candidates,
            )

        for parameter in parameter_space:
            for value in parameter.values:
                candidate = dict(anchor_parameters)
                candidate[parameter.name] = value
                if not self._is_valid_parameter_combination(candidate):
                    continue
                self._append_candidate_if_new(
                    parameter_space=parameter_space,
                    candidate=candidate,
                    seen=seen,
                    candidates=candidates,
                )

        for left_index, left_parameter in enumerate(parameter_space):
            for right_parameter in parameter_space[left_index + 1 :]:
                pair_budget = min(
                    len(left_parameter.values) * len(right_parameter.values),
                    config.initial_samples,
                )
                pair_candidates = 0
                for left_value in left_parameter.values:
                    for right_value in right_parameter.values:
                        candidate = dict(anchor_parameters)
                        candidate[left_parameter.name] = left_value
                        candidate[right_parameter.name] = right_value
                        if not self._is_valid_parameter_combination(candidate):
                            continue
                        self._append_candidate_if_new(
                            parameter_space=parameter_space,
                            candidate=candidate,
                            seen=seen,
                            candidates=candidates,
                        )
                        pair_candidates += 1
                        if pair_candidates >= pair_budget or len(candidates) >= target:
                            break
                    if pair_candidates >= pair_budget or len(candidates) >= target:
                        break
                if len(candidates) >= target:
                    break

        if len(candidates) < target:
            random_fill = self._sample_parameter_sets(
                parameter_space,
                target - len(candidates),
                rng,
                seen,
            )
            for parameters in random_fill:
                if not self._is_valid_parameter_combination(parameters):
                    continue
                self._append_candidate_if_new(
                    parameter_space=parameter_space,
                    candidate=parameters,
                    seen=seen,
                    candidates=candidates,
                )

        logger.info(
            "adaptive_search_mode mode=hybrid total_combinations=%s initial_candidates=%s target=%s",
            total,
            len(candidates),
            target,
        )
        return candidates

    def _sample_parameter_sets(
        self,
        parameter_space: Sequence[ParameterRange],
        sample_size: int,
        rng: random.Random,
        excluded: set[tuple[Any, ...]],
    ) -> list[dict[str, Any]]:
        total = self._total_combinations(parameter_space)
        if total <= sample_size + len(excluded):
            sampled: list[dict[str, Any]] = []
            for values in product(*(parameter.values for parameter in parameter_space)):
                parameters = self._parameter_dict(parameter_space, values)
                if not self._is_valid_parameter_combination(parameters):
                    continue
                key = self._parameter_key(parameter_space, parameters)
                if key not in excluded:
                    sampled.append(parameters)
            return sampled

        sampled: list[dict[str, Any]] = []
        seen = set(excluded)
        max_attempts = max(sample_size * 20, 2_000)
        attempts = 0
        while len(sampled) < sample_size and attempts < max_attempts:
            attempts += 1
            parameters = {
                parameter.name: rng.choice(parameter.values)
                for parameter in parameter_space
            }
            if not self._is_valid_parameter_combination(parameters):
                continue
            key = self._parameter_key(parameter_space, parameters)
            if key in seen:
                continue
            seen.add(key)
            sampled.append(parameters)

        if len(sampled) < sample_size:
            logger.warning(
                "adaptive_search_sampling_shortfall requested=%s sampled=%s total=%s excluded=%s attempts=%s",
                sample_size,
                len(sampled),
                total,
                len(excluded),
                attempts,
            )
        return sampled

    def _neighbor_parameter_sets(
        self,
        parameter_space: Sequence[ParameterRange],
        base_parameters: dict[str, Any],
        excluded: set[tuple[Any, ...]],
    ) -> list[dict[str, Any]]:
        neighbors: list[dict[str, Any]] = []
        seen = set(excluded)

        for parameter in parameter_space:
            values = parameter.values
            if len(values) <= 1:
                continue
            current_value = base_parameters[parameter.name]
            current_index = values.index(current_value)
            lower_index = max(0, current_index - self.search_config.neighbor_window)
            upper_index = min(len(values), current_index + self.search_config.neighbor_window + 1)
            for neighbor_index in range(lower_index, upper_index):
                if neighbor_index == current_index:
                    continue
                neighbor_parameters = dict(base_parameters)
                neighbor_parameters[parameter.name] = values[neighbor_index]
                if not self._is_valid_parameter_combination(neighbor_parameters):
                    continue
                key = self._parameter_key(parameter_space, neighbor_parameters)
                if key in seen:
                    continue
                seen.add(key)
                neighbors.append(neighbor_parameters)
        return neighbors

    def evaluate(
        self,
        *,
        strategy_name: str,
        parameters: dict[str, Any],
        strategy_factory: StrategyFactory,
    ) -> TuningResult | None:
        strategy = strategy_factory(parameters)
        logger.debug(
            "tuner_evaluate strategy=%s parameters=%s",
            strategy_name,
            self._format_parameters(parameters),
        )
        seed_brokerage = self.brokerage_factory()
        backtester = Backtester(
            strategy=strategy,
            brokerage=seed_brokerage,
            asset_name=self.asset_name,
            context_orchestrator=self.context_orchestrator,
        )

        daily_results = backtester.simulate_by_day(
            brokerage=seed_brokerage,
            strategy=strategy,
            brokerage_factory=self.brokerage_factory,
        )

        balances = [result.end_balance for result in daily_results]
        labels = [
            result.session_date if result.session_date is not None else str(index + 1)
            for index, result in enumerate(daily_results)
        ]
        if not balances:
            return None

        equity_curve = [self.initial_balance, *[float(balance) for balance in balances]]
        returns = returns_from_equity_curve(equity_curve)
        if not returns:
            return None

        metrics = summarize_risk_metrics(returns=returns, equity_curve=equity_curve)
        return TuningResult(
            strategy_name=strategy_name,
            parameters=dict(parameters),
            balances=balances,
            labels=labels,
            sharpe=metrics["sharpe"],
            sortino=metrics["sortino"],
            calmar=metrics["calmar"],
            max_drawdown=metrics["max_drawdown"],
            final_balance=balances[-1],
        )

    def adaptive_search(
        self,
        *,
        strategy_name: str,
        parameter_space: Sequence[ParameterRange],
        strategy_factory: StrategyFactory,
    ) -> list[TuningResult]:
        config = self.search_config
        rng = random.Random(config.random_seed)
        evaluated: dict[tuple[Any, ...], TuningResult] = {}

        parameter_space = self._effective_parameter_space(parameter_space)
        logger.info(
            "adaptive_search_effective_parameter_space strategy=%s dimensions=%s",
            strategy_name,
            {parameter.name: len(parameter.values) for parameter in parameter_space},
        )

        expanded_parameter_space = self._expand_parameter_space(parameter_space, config)
        if expanded_parameter_space != list(parameter_space):
            logger.info(
                "adaptive_search_parameter_space_expanded strategy=%s original_total=%s expanded_total=%s original_dimensions=%s expanded_dimensions=%s",
                strategy_name,
                self._total_combinations(parameter_space),
                self._total_combinations(expanded_parameter_space),
                {parameter.name: len(parameter.values) for parameter in parameter_space},
                {parameter.name: len(parameter.values) for parameter in expanded_parameter_space},
            )
        parameter_space = expanded_parameter_space

        initial_candidates = self._build_initial_candidates(
            parameter_space,
            config,
            rng,
        )
        logger.info(
            "adaptive_search_start strategy=%s initial_candidates=%s total_combinations=%s",
            strategy_name,
            len(initial_candidates),
            self._total_combinations(parameter_space),
        )
        for index, parameters in enumerate(initial_candidates, start=1):
            logger.debug(
                "adaptive_search_initial_candidate strategy=%s index=%s parameters=%s",
                strategy_name,
                index,
                self._format_parameters(parameters),
            )

        initial_results = self._evaluate_parameter_batch_parallel(
            strategy_name=strategy_name,
            parameter_space=parameter_space,
            parameter_sets=initial_candidates,
            strategy_factory=strategy_factory,
            round_label="initial",
        )
        for key, result in initial_results.items():
            logger.debug(
                "adaptive_search_initial_result strategy=%s parameters=%s sharpe=%.6f sortino=%.6f calmar=%.6f final_balance=%.2f",
                strategy_name,
                self._format_parameters(result.parameters),
                result.sharpe,
                result.sortino,
                result.calmar,
                result.final_balance,
            )
            evaluated[key] = result

        for round_index in range(config.refinement_rounds):
            if not evaluated:
                break

            ranked = sorted(evaluated.values(), key=self._score_result, reverse=True)
            top_results = ranked[: min(config.top_k, len(ranked))]
            candidate_parameters: list[dict[str, Any]] = []
            excluded = set(evaluated.keys())

            for result in top_results:
                neighbors = self._neighbor_parameter_sets(
                    parameter_space,
                    result.parameters,
                    excluded,
                )
                candidate_parameters.extend(neighbors)
                excluded.update(
                    self._parameter_key(parameter_space, parameters)
                    for parameters in neighbors
                )

            exploratory_budget = min(
                config.exploratory_samples_per_round + round_index * config.top_k * config.round_random_multiplier,
                max(self._total_combinations(parameter_space) - len(excluded), 0),
            )
            exploratory = self._sample_parameter_sets(
                parameter_space,
                exploratory_budget,
                rng,
                excluded,
            )
            candidate_parameters.extend(exploratory)
            candidate_parameters = self._dedupe_parameter_sets(parameter_space, candidate_parameters)

            if not candidate_parameters:
                logger.info(
                    "adaptive_search_stopped strategy=%s round=%s reason=no_new_candidates",
                    strategy_name,
                    round_index + 1,
                )
                break

            logger.info(
                "adaptive_search_round strategy=%s round=%s top_results=%s neighbor_candidates=%s exploratory_candidates=%s total_round_candidates=%s excluded=%s remaining=%s",
                strategy_name,
                round_index + 1,
                len(top_results),
                len(candidate_parameters) - len(exploratory),
                len(exploratory),
                len(candidate_parameters),
                len(excluded),
                max(self._total_combinations(parameter_space) - len(excluded), 0),
            )
            for index, parameters in enumerate(candidate_parameters, start=1):
                logger.debug(
                    "adaptive_search_round_candidate strategy=%s round=%s index=%s parameters=%s",
                    strategy_name,
                    round_index + 1,
                    index,
                    self._format_parameters(parameters),
                )

            round_results = self._evaluate_parameter_batch_parallel(
                strategy_name=strategy_name,
                parameter_space=parameter_space,
                parameter_sets=candidate_parameters,
                strategy_factory=strategy_factory,
                round_label=str(round_index + 1),
            )
            for key, result in round_results.items():
                logger.debug(
                    "adaptive_search_round_result strategy=%s round=%s parameters=%s sharpe=%.6f sortino=%.6f calmar=%.6f final_balance=%.2f",
                    strategy_name,
                    round_index + 1,
                    self._format_parameters(result.parameters),
                    result.sharpe,
                    result.sortino,
                    result.calmar,
                    result.final_balance,
                )
                evaluated[key] = result

        ranked_results = sorted(evaluated.values(), key=self._score_result, reverse=True)
        logger.info(
            "adaptive_search_complete strategy=%s evaluated=%s unique_parameter_sets=%s total_combinations=%s coverage_pct=%.2f executor=%s workers=%s configured_parallelism=%s configured_workers=%s best_parameters=%s",
            strategy_name,
            len(ranked_results),
            len(evaluated),
            self._total_combinations(parameter_space),
            100.0 * len(evaluated) / max(self._total_combinations(parameter_space), 1),
            "thread" if self.search_config.executor_kind == "process" else self.search_config.executor_kind,
            self._resolved_max_parallel_workers(),
            self.search_config.max_parallelism,
            self.search_config.max_parallel_workers,
            self._format_parameters(ranked_results[0].parameters) if ranked_results else "none",
        )
        return ranked_results

    @staticmethod
    def summarize_best_results(results: Sequence[TuningResult]) -> dict[str, dict[str, object]]:
        if not results:
            raise ValueError("results must not be empty")

        best_sharpe = max(results, key=lambda result: result.sharpe)
        best_sortino = max(results, key=lambda result: result.sortino)
        best_calmar = max(results, key=lambda result: result.calmar)
        best_final_balance = max(results, key=lambda result: result.final_balance)

        sharpe_ranks = {
            id(result): rank
            for rank, result in enumerate(
                sorted(results, key=lambda item: item.sharpe, reverse=True),
                start=1,
            )
        }
        sortino_ranks = {
            id(result): rank
            for rank, result in enumerate(
                sorted(results, key=lambda item: item.sortino, reverse=True),
                start=1,
            )
        }
        calmar_ranks = {
            id(result): rank
            for rank, result in enumerate(
                sorted(results, key=lambda item: item.calmar, reverse=True),
                start=1,
            )
        }
        balance_ranks = {
            id(result): rank
            for rank, result in enumerate(
                sorted(results, key=lambda item: item.final_balance, reverse=True),
                start=1,
            )
        }

        best_composite = min(
            results,
            key=lambda result: (
                sharpe_ranks[id(result)]
                + sortino_ranks[id(result)]
                + calmar_ranks[id(result)]
                + balance_ranks[id(result)]
            ),
        )

        def pack(result: TuningResult) -> dict[str, object]:
            return {
                "strategy_name": result.strategy_name,
                "parameters": result.parameters,
                "sharpe": result.sharpe,
                "sortino": result.sortino,
                "calmar": result.calmar,
                "max_drawdown": result.max_drawdown,
                "final_balance": result.final_balance,
            }

        def pack_with_rank(
            result: TuningResult,
            *,
            sharpe_rank: int,
            sortino_rank: int,
            calmar_rank: int,
            balance_rank: int,
        ) -> dict[str, object]:
            packed = pack(result)
            packed["ranks"] = {
                "sharpe": sharpe_rank,
                "sortino": sortino_rank,
                "calmar": calmar_rank,
                "final_balance": balance_rank,
            }
            return packed

        return {
            "best_sharpe": pack_with_rank(
                best_sharpe,
                sharpe_rank=sharpe_ranks[id(best_sharpe)],
                sortino_rank=sortino_ranks[id(best_sharpe)],
                calmar_rank=calmar_ranks[id(best_sharpe)],
                balance_rank=balance_ranks[id(best_sharpe)],
            ),
            "best_sortino": pack_with_rank(
                best_sortino,
                sharpe_rank=sharpe_ranks[id(best_sortino)],
                sortino_rank=sortino_ranks[id(best_sortino)],
                calmar_rank=calmar_ranks[id(best_sortino)],
                balance_rank=balance_ranks[id(best_sortino)],
            ),
            "best_calmar": pack_with_rank(
                best_calmar,
                sharpe_rank=sharpe_ranks[id(best_calmar)],
                sortino_rank=sortino_ranks[id(best_calmar)],
                calmar_rank=calmar_ranks[id(best_calmar)],
                balance_rank=balance_ranks[id(best_calmar)],
            ),
            "best_final_balance": pack_with_rank(
                best_final_balance,
                sharpe_rank=sharpe_ranks[id(best_final_balance)],
                sortino_rank=sortino_ranks[id(best_final_balance)],
                calmar_rank=calmar_ranks[id(best_final_balance)],
                balance_rank=balance_ranks[id(best_final_balance)],
            ),
            "best_composite": pack_with_rank(
                best_composite,
                sharpe_rank=sharpe_ranks[id(best_composite)],
                sortino_rank=sortino_ranks[id(best_composite)],
                calmar_rank=calmar_ranks[id(best_composite)],
                balance_rank=balance_ranks[id(best_composite)],
            ),
        }
