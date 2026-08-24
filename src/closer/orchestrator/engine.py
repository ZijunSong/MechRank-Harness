"""CLOSER orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

from closer import TRACE_SCHEMA_VERSION
from closer.assay.interpreter import AssayInterpreter, deterministic_ablation_contract
from closer.audit.auditor import AdaptiveAuditor
from closer.benchmark.pgllm_renderer import SYSTEM_PROMPT, render_user_prompt
from closer.benchmark.validators import validate_final_ranking
from closer.comparison.reasoner import ComparativeReasoner
from closer.comparison.scheduler import ComparisonScheduler, bridge_pairs
from closer.config import CloserConfig
from closer.errors import GraphConnectivityError
from closer.evidence.analyzer import MechanisticAnalyzer, placeholder_uncertain_states
from closer.evidence.state import EpisodeEvidenceState
from closer.llm.base import LLMClient
from closer.logging import RunStore, append_jsonl, dump_json, sha256_text
from closer.metrics.closure import collect_diagnostics
from closer.metrics.efficiency import efficiency_snapshot
from closer.mutation.compiler import compile_episode_mutations
from closer.orchestrator.budget import BudgetManager
from closer.ranking.bradley_terry import borda_from_setwise
from closer.ranking.graph import PreferenceGraph
from closer.ranking.solver import GlobalRankSolver
from closer.schemas.episode import ProteinEpisode
from closer.schemas.ranking import CloserResult, RankingJSON


class CloserEngine:
    def __init__(
        self,
        config: CloserConfig,
        client: LLMClient,
        *,
        store: RunStore | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.store = store
        self.scheduler = ComparisonScheduler(
            group_size=config.comparison.group_size,
            overlap=config.comparison.overlap,
        )
        self.solver = GlobalRankSolver(config.ranking)

    async def run_episode(
        self,
        episode: ProteinEpisode,
        *,
        max_output_tokens: int | None = None,
    ) -> CloserResult:
        budget = BudgetManager(self.config.budget)
        episode_id = sha256_text(
            episode.protein_name + episode.assay_description + "".join(episode.variant_ids())
        )[:16]
        ep_dir = self.store.episode_dir(episode_id) if self.store else None
        if ep_dir is not None:
            self.client.trace_dir = ep_dir / "llm_calls"  # type: ignore[attr-defined]
            dump_json(ep_dir / "input.json", {"schema_version": TRACE_SCHEMA_VERSION, **episode.model_dump()})

        if self.config.ablation.mutation_compiler:
            profiles = compile_episode_mutations(episode)
        else:
            from closer.schemas.mutation import VariantMutationProfile

            profiles = {
                variant.variant_id: VariantMutationProfile(
                    variant_id=variant.variant_id, substitutions=[], mutation_count=0
                )
                for variant in episode.variants
            }
        if ep_dir is not None:
            dump_json(
                ep_dir / "mutation_profiles.json",
                {k: v.model_dump(mode="json") for k, v in profiles.items()},
            )

        if self.config.ablation.assay_contract and self.config.assay.enabled:
            contract = await AssayInterpreter(self.client).interpret(episode, budget=budget)
        else:
            contract = deterministic_ablation_contract(episode)
        if ep_dir is not None:
            dump_json(ep_dir / "assay_contract.json", contract.model_dump(mode="json"))

        if self.config.ablation.evidence_state and self.config.evidence.enabled:
            analyzer = MechanisticAnalyzer(
                self.client,
                max_variants_per_call=self.config.evidence.max_variants_per_call,
                parallelism=self.config.evidence.parallelism,
            )
            evidence = await analyzer.analyze_episode(
                episode,
                contract,
                profiles,
                budget=budget,
                evidence_dir=ep_dir / "evidence" if ep_dir else None,
                window=self.config.evidence.local_context_window,
            )
        else:
            evidence = placeholder_uncertain_states(profiles)
        state = EpisodeEvidenceState(
            episode=episode,
            assay_contract=contract,
            mutation_profiles=profiles,
            variant_evidence=evidence,
        )
        if ep_dir is not None:
            state.save_json(ep_dir / "evidence_states.json")

        if not self.config.ablation.comparative_reasoning:
            result = await self._direct_rank_path(
                episode,
                state,
                budget,
                ep_dir,
                max_output_tokens=max_output_tokens,
            )
            return result

        graph = PreferenceGraph()
        graph.add_nodes(episode.variant_ids())
        reasoner = ComparativeReasoner(self.client)
        groups = self.scheduler.initial_groups(evidence, episode.variant_ids())
        setwise_rankings = await self._run_setwise(
            reasoner, state, groups, graph, round_index=0, budget=budget, ep_dir=ep_dir
        )
        boundary = self.scheduler.boundary_groups(groups)
        setwise_rankings.extend(
            await self._run_setwise(
                reasoner, state, boundary, graph, round_index=1, budget=budget, ep_dir=ep_dir
            )
        )

        await self._ensure_connected(reasoner, state, graph, episode.variant_ids(), budget, ep_dir)

        if self.config.ablation.global_solver:
            before = self.solver.solve(episode.variant_ids(), graph)
        else:
            before = borda_from_setwise(episode.variant_ids(), setwise_rankings)
        if ep_dir is not None:
            dump_json(ep_dir / "solver_before_audit.json", before.model_dump(mode="json"))
            dump_json(ep_dir / "graph.json", graph.dump())

        cycle_before = len(graph.find_cycles())
        audit_history: list[dict] = []
        after = before
        if self.config.ablation.adaptive_audit and self.config.audit.enabled:
            auditor = AdaptiveAuditor(self.client, self.config.audit, self.solver)
            after, audit_history = await auditor.run(
                variant_ids=episode.variant_ids(),
                state=state,
                graph=graph,
                provisional=before,
                budget=budget,
            )
            if ep_dir is not None:
                for row in audit_history:
                    append_jsonl(ep_dir / "comparisons" / "audit_rounds.jsonl", row)
        elif self.config.ablation.global_solver:
            after = self.solver.solve(episode.variant_ids(), graph)

        ranking = validate_final_ranking(episode, after.ranking)
        usage = efficiency_snapshot(budget)
        diagnostics = collect_diagnostics(
            evidence=evidence,
            graph=graph,
            before=before,
            after=after,
            audit_history=audit_history,
            usage=usage,
            cycle_count_before=cycle_before,
            config=self.config.audit,
        )
        result = CloserResult(
            ranking=ranking,
            scores=after.scores,
            assay_contract=contract,
            diagnostics=diagnostics,
            usage=usage,
            trace_path=str(ep_dir) if ep_dir else "",
        )
        if ep_dir is not None:
            dump_json(ep_dir / "solver_after_audit.json", after.model_dump(mode="json"))
            dump_json(ep_dir / "diagnostics.json", diagnostics)
            dump_json(ep_dir / "final_result.json", result.model_dump(mode="json"))
            dump_json(ep_dir / "graph.json", graph.dump())
        return result

    async def _run_setwise(
        self,
        reasoner: ComparativeReasoner,
        state: EpisodeEvidenceState,
        groups: list[list[str]],
        graph: PreferenceGraph,
        *,
        round_index: int,
        budget,
        ep_dir: Path | None,
    ) -> list[list[str]]:
        rankings: list[list[str]] = []
        for group in groups:
            result = await reasoner.rank_set(state, group, budget=budget)
            graph.add_setwise(result, round_index=round_index)
            rankings.append(result.ranking)
            if ep_dir is not None:
                append_jsonl(
                    ep_dir / "comparisons" / f"round_{round_index:02d}.jsonl",
                    result.model_dump(mode="json"),
                )
        return rankings

    async def _ensure_connected(
        self,
        reasoner: ComparativeReasoner,
        state: EpisodeEvidenceState,
        graph: PreferenceGraph,
        variant_ids: list[str],
        budget,
        ep_dir: Path | None,
    ) -> None:
        attempts = 0
        while not graph.is_weakly_connected():
            attempts += 1
            if attempts > len(variant_ids):
                raise GraphConnectivityError("could not connect preference graph with bridge comparisons")
            components = graph.connected_components()
            for left, right in bridge_pairs(components):
                pref = await reasoner.compare_pair(state, left, right, budget=budget)
                graph.add_pair_preference(pref, source="pairwise", round_index=50 + attempts)
                if ep_dir is not None:
                    append_jsonl(
                        ep_dir / "comparisons" / "bridges.jsonl",
                        pref.model_dump(mode="json"),
                    )
            if graph.is_weakly_connected():
                return
        return

    async def _direct_rank_path(
        self,
        episode: ProteinEpisode,
        state: EpisodeEvidenceState,
        budget,
        ep_dir: Path | None,
        *,
        max_output_tokens: int | None = None,
    ) -> CloserResult:
        user = episode.raw_user_prompt or render_user_prompt(episode)
        extras = []
        if self.config.ablation.mutation_compiler:
            extras.append(
                "Deterministic mutation profiles:\n"
                + json.dumps(
                    {k: v.shorthand() for k, v in state.mutation_profiles.items()},
                    indent=2,
                )
            )
        if self.config.ablation.assay_contract:
            extras.append("Assay contract JSON:\n" + state.assay_contract.model_dump_json(indent=2))
        if self.config.ablation.evidence_state:
            extras.append(
                "Structured evidence:\n"
                + json.dumps(
                    {k: v.model_dump(mode="json") for k, v in state.variant_evidence.items()},
                    indent=2,
                )
            )
        if extras:
            user = user + "\n\nAdditional harness context (Track A, no external tools):\n" + "\n\n".join(extras)
        parsed, _trace = await self.client.generate_structured(
            stage="direct_rank",
            schema=RankingJSON,
            system=episode.system_prompt or SYSTEM_PROMPT,
            user=user,
            budget=budget,
            max_output_tokens=max_output_tokens,
        )
        ranking = validate_final_ranking(episode, parsed.ranking)
        scores = {vid: float(len(ranking) - i) for i, vid in enumerate(ranking)}
        usage = efficiency_snapshot(budget)
        diagnostics = {"mode": "direct_rank", **usage}
        result = CloserResult(
            ranking=ranking,
            scores=scores,
            assay_contract=state.assay_contract,
            diagnostics=diagnostics,
            usage=usage,
            trace_path=str(ep_dir) if ep_dir else "",
        )
        if ep_dir is not None:
            dump_json(ep_dir / "final_result.json", result.model_dump(mode="json"))
            dump_json(ep_dir / "diagnostics.json", diagnostics)
        return result
