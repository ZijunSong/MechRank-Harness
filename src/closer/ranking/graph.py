"""Preference graph over variant ids. Raw observations are append-only."""

from __future__ import annotations

import uuid
from collections import defaultdict

import networkx as nx

from closer.comparison.validation import canonical_pair, validate_setwise_pair_preferences
from closer.errors import ComparisonError
from closer.schemas.comparison import (
    PairPreference,
    PreferenceEdge,
    PreferenceObservation,
    SetwiseRanking,
)


class PreferenceGraph:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.observations: list[PreferenceObservation] = []

    def add_nodes(self, variant_ids: list[str]) -> None:
        self.graph.add_nodes_from(variant_ids)

    def raw_observations(self) -> list[PreferenceObservation]:
        return list(self.observations)

    def active_observations(self) -> list[PreferenceObservation]:
        return [obs for obs in self.observations if obs.status == "active"]

    def add_observation(self, observation: PreferenceObservation) -> PreferenceObservation:
        if not observation.observation_id:
            observation = observation.model_copy(update={"observation_id": uuid.uuid4().hex[:12]})
        self._reject_unknown_endpoints(observation.left_id, observation.right_id)
        if observation.source == "audit":
            observation = self._supersede_same_pair(observation)
        self.observations.append(observation)
        self._rebuild_effective_graph()
        return observation

    def add_pair_preference(
        self,
        pref: PairPreference,
        *,
        source: str,
        round_index: int,
        comparison_event_id: str | None = None,
        derived_from_order: bool = False,
    ) -> PreferenceObservation:
        obs = PreferenceObservation(
            observation_id=uuid.uuid4().hex[:12],
            left_id=pref.left_id,
            right_id=pref.right_id,
            winner=pref.winner,
            confidence=pref.confidence,
            source=source,  # type: ignore[arg-type]
            round_index=round_index,
            rationale=pref.rationale,
            key_evidence=pref.key_evidence,
            contradiction_with_previous_evidence=pref.contradiction_with_previous_evidence,
            comparison_event_id=comparison_event_id,
            derived_from_order=derived_from_order,
        )
        return self.add_observation(obs)

    def add_setwise(self, result: SetwiseRanking, *, round_index: int) -> None:
        event_id = uuid.uuid4().hex[:12]
        group_ids = list(result.variant_ids) or list(result.ranking)
        explicit = validate_setwise_pair_preferences(result.pair_preferences, group_ids)
        explicit_keys = {canonical_pair(pref.left_id, pref.right_id) for pref in explicit}
        for pref in explicit:
            self.add_pair_preference(
                pref,
                source="setwise",
                round_index=round_index,
                comparison_event_id=event_id,
            )
        ranking = result.ranking
        for winner, loser in zip(ranking, ranking[1:], strict=False):
            key = canonical_pair(winner, loser)
            if key in explicit_keys:
                continue
            self.add_observation(
                PreferenceObservation(
                    observation_id=uuid.uuid4().hex[:12],
                    left_id=winner,
                    right_id=loser,
                    winner="left",
                    confidence=result.confidence,
                    source="setwise",
                    round_index=round_index,
                    rationale=f"setwise adjacent derived from order: {winner} > {loser}",
                    comparison_event_id=event_id,
                    derived_from_order=True,
                )
            )

    def aggregate_pair(self, id1: str, id2: str) -> dict:
        relevant = [
            obs
            for obs in self.active_observations()
            if {obs.left_id, obs.right_id} == {id1, id2} and obs.winner in {"left", "right"}
        ]
        votes: dict[str, float] = defaultdict(float)
        for obs in relevant:
            winner = obs.left_id if obs.winner == "left" else obs.right_id
            votes[winner] += obs.confidence
        return {
            "pair": tuple(sorted((id1, id2))),
            "n_observations": len(relevant),
            "votes": dict(votes),
        }

    def find_cycles(self, cutoff: int = 20) -> list[list[str]]:
        cycles = []
        for cycle in nx.simple_cycles(self.graph):
            cycles.append(cycle)
            if len(cycles) >= cutoff:
                break
        return cycles

    def pair_margin(self, scores: dict[str, float], id1: str, id2: str) -> float:
        return abs(scores[id1] - scores[id2])

    def uncertain_pairs(self) -> list[tuple[str, str]]:
        pairs = []
        for obs in self.active_observations():
            if obs.winner in {"tie", "uncertain"}:
                pairs.append(tuple(sorted((obs.left_id, obs.right_id))))
        return sorted(set(pairs))

    def connected_components(self) -> list[list[str]]:
        undirected = self.graph.to_undirected()
        return [sorted(component) for component in nx.connected_components(undirected)]

    def is_weakly_connected(self) -> bool:
        if self.graph.number_of_nodes() == 0:
            return False
        return nx.is_weakly_connected(self.graph)

    def informative_edges(self) -> list[PreferenceEdge]:
        edges: list[PreferenceEdge] = []
        for obs in self.active_observations():
            if obs.winner not in {"left", "right"}:
                continue
            winner = obs.left_id if obs.winner == "left" else obs.right_id
            loser = obs.right_id if obs.winner == "left" else obs.left_id
            edges.append(
                PreferenceEdge(
                    winner=winner,
                    loser=loser,
                    confidence=obs.confidence,
                    source=obs.source,
                    round_index=obs.round_index,
                    rationale_ref=obs.observation_id,
                    observation_id=obs.observation_id,
                )
            )
        return edges

    def dump(self) -> dict:
        return {
            "nodes": list(self.graph.nodes),
            "observations": [obs.model_dump(mode="json") for obs in self.observations],
            "active_observations": [obs.model_dump(mode="json") for obs in self.active_observations()],
            "cycles": self.find_cycles(),
            "components": self.connected_components(),
        }

    def _reject_unknown_endpoints(self, left_id: str, right_id: str) -> None:
        known = set(self.graph.nodes)
        if not known:
            return
        if left_id not in known or right_id not in known:
            raise ComparisonError(
                f"unknown variant id in observation: left={left_id!r} right={right_id!r}"
            )

    def _supersede_same_pair(self, incoming: PreferenceObservation) -> PreferenceObservation:
        key = canonical_pair(incoming.left_id, incoming.right_id)
        last_id: str | None = None
        updated: list[PreferenceObservation] = []
        for obs in self.observations:
            if (
                obs.status == "active"
                and canonical_pair(obs.left_id, obs.right_id) == key
            ):
                last_id = obs.observation_id
                updated.append(obs.model_copy(update={"status": "superseded"}))
            else:
                updated.append(obs)
        self.observations = updated
        if last_id and not incoming.revision_of:
            return incoming.model_copy(update={"revision_of": last_id})
        return incoming

    def _rebuild_effective_graph(self) -> None:
        nodes = list(self.graph.nodes)
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(nodes)
        for obs in self.active_observations():
            if obs.winner not in {"left", "right"}:
                continue
            winner = obs.left_id if obs.winner == "left" else obs.right_id
            loser = obs.right_id if obs.winner == "left" else obs.left_id
            self.graph.add_edge(
                winner,
                loser,
                confidence=obs.confidence,
                source=obs.source,
                observation_id=obs.observation_id,
            )
