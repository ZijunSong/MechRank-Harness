"""Preference graph over variant ids. Observations are append-only."""

from __future__ import annotations

import uuid
from collections import defaultdict

import networkx as nx

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

    def add_observation(self, observation: PreferenceObservation) -> PreferenceObservation:
        if observation.observation_id is None:
            observation = observation.model_copy(update={"observation_id": uuid.uuid4().hex[:12]})
        self.observations.append(observation)
        if observation.winner in {"left", "right"}:
            winner = observation.left_id if observation.winner == "left" else observation.right_id
            loser = observation.right_id if observation.winner == "left" else observation.left_id
            self.graph.add_edge(
                winner,
                loser,
                confidence=observation.confidence,
                source=observation.source,
                observation_id=observation.observation_id,
            )
        return observation

    def add_pair_preference(
        self,
        pref: PairPreference,
        *,
        source: str,
        round_index: int,
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
        )
        return self.add_observation(obs)

    def add_setwise(self, result: SetwiseRanking, *, round_index: int) -> None:
        ranking = result.ranking
        for winner, loser in zip(ranking, ranking[1:], strict=False):
            self.add_observation(
                PreferenceObservation(
                    observation_id=uuid.uuid4().hex[:12],
                    left_id=winner,
                    right_id=loser,
                    winner="left",
                    confidence=result.confidence,
                    source="setwise",
                    round_index=round_index,
                    rationale=f"setwise adjacent: {winner} > {loser}",
                )
            )
        for pref in result.pair_preferences:
            self.add_pair_preference(pref, source="setwise", round_index=round_index)

    def aggregate_pair(self, id1: str, id2: str) -> dict:
        relevant = [
            obs
            for obs in self.observations
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
        for obs in self.observations:
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
        for obs in self.observations:
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
            "cycles": self.find_cycles(),
            "components": self.connected_components(),
        }
