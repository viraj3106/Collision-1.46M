"""
COLLISION Synaptic Working Memory & Global Workspace (GWT-SWM).

Implements Bernard Baars & Stanislas Dehaene's Global Workspace Theory adapted for
neuro-symbolic language models:
- Unconscious parallel cognitive modules compete for conscious access
- Central Global Workspace broadcasts the winning hypothesis
- Synaptic Working Memory tracks cognitive items with temporal decay, activation boosting, and associative recall
"""

import time
import math
from typing import Dict, Any, List, Optional, Tuple, Callable
from pydantic import BaseModel, Field

from collision.brain.schemas import (
    GlobalWorkspaceMessage,
    WorkingMemoryItem,
    CognitiveModality
)


class SynapticWorkingMemory:
    """
    Dynamic Working Memory buffer simulating biological synaptic plasticity:
    - Items have salience (activation potential) that decays over time.
    - Repeated access strengthens salience (LTP - Long Term Potentiation).
    - Capacity is bounded (Miller's Law 7±2 active items in working focus).
    """

    def __init__(self, capacity: int = 9, half_life_seconds: float = 300.0):
        self.capacity = capacity
        self.half_life_seconds = half_life_seconds
        self.items: Dict[str, WorkingMemoryItem] = {}

    def retain(self, key: str, value: Any, salience: float = 1.0, domain_tags: Optional[List[str]] = None) -> WorkingMemoryItem:
        """Stores or reactivates an item in working memory."""
        now = time.time()
        self._apply_decay(now)

        if key in self.items:
            item = self.items[key]
            item.value = value
            item.salience = min(2.0, item.salience + salience * 0.5)  # LTP reinforcement
            item.access_count += 1
            item.last_accessed = now
            if domain_tags:
                for tag in domain_tags:
                    if tag not in item.domain_tags:
                        item.domain_tags.append(tag)
        else:
            item = WorkingMemoryItem(
                key=key,
                value=value,
                salience=salience,
                decay_rate=math.log(2) / max(1.0, self.half_life_seconds),
                access_count=1,
                last_accessed=now,
                domain_tags=domain_tags or []
            )
            self.items[key] = item

        self._prune_if_needed()
        return item

    def recall(self, key: str) -> Optional[Any]:
        """Retrieves and boosts activation of an item."""
        now = time.time()
        self._apply_decay(now)
        if key in self.items:
            item = self.items[key]
            item.last_accessed = now
            item.access_count += 1
            item.salience = min(2.0, item.salience + 0.2)
            return item.value
        return None

    def search_by_tag(self, tag: str) -> List[Tuple[str, Any, float]]:
        """Associative recall by domain tag."""
        now = time.time()
        self._apply_decay(now)
        results = []
        for k, v in self.items.items():
            if tag.lower() in [t.lower() for t in v.domain_tags]:
                results.append((k, v.value, v.salience))
        return sorted(results, key=lambda x: x[2], reverse=True)

    def _apply_decay(self, current_time: float) -> None:
        """Applies exponential decay to all items."""
        dead_keys = []
        for k, item in self.items.items():
            dt = max(0.0, current_time - item.last_accessed)
            decayed = item.salience * math.exp(-item.decay_rate * dt)
            item.salience = decayed
            if decayed < 0.05:
                dead_keys.append(k)
        for dk in dead_keys:
            del self.items[dk]

    def _prune_if_needed(self) -> None:
        """Retains only the top-K salient items according to capacity."""
        if len(self.items) > self.capacity:
            sorted_items = sorted(self.items.items(), key=lambda x: x[1].salience, reverse=True)
            self.items = dict(sorted_items[:self.capacity])

    def snapshot(self) -> Dict[str, Any]:
        """Returns a readable snapshot of active working memory items."""
        now = time.time()
        self._apply_decay(now)
        return {
            k: {
                "value": v.value,
                "salience": round(v.salience, 3),
                "access_count": v.access_count,
                "domain_tags": v.domain_tags
            }
            for k, v in self.items.items()
        }


class GlobalWorkspace:
    """
    Central conscious blackboard coordinating competing unconscious specialized modules:
    - Modules submit candidate proposals with salience scores.
    - The Workspace arbitrates and selects the most salient consensus.
    - Broadcasts the winning state across all modules for coherent unified output.
    """

    def __init__(self):
        self.working_memory = SynapticWorkingMemory()
        self.broadcast_history: List[GlobalWorkspaceMessage] = []
        self.cycle_counter: int = 0

    def broadcast(
        self,
        sender_module: str,
        salience_weight: float,
        content_summary: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> GlobalWorkspaceMessage:
        """Broadcasts a high-salience message to the workspace."""
        self.cycle_counter += 1
        msg = GlobalWorkspaceMessage(
            sender_module=sender_module,
            salience_weight=salience_weight,
            content_summary=content_summary,
            payload=payload or {},
            broadcast_cycle=self.cycle_counter,
            timestamp=time.time()
        )
        self.broadcast_history.append(msg)
        # Retain summary in working memory
        self.working_memory.retain(
            key=f"broadcast_{sender_module}_{self.cycle_counter}",
            value=content_summary,
            salience=salience_weight,
            domain_tags=[sender_module]
        )
        return msg

    def arbitrate_candidates(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Selects the winning candidate from multiple competing specialized modules
        based on confidence, salience, and epistemic consistency.
        """
        if not candidates:
            return None

        # Sort candidates by combined score = confidence * 0.6 + salience * 0.4
        def score_cand(c: Dict[str, Any]) -> float:
            conf = float(c.get("confidence", 0.5))
            sal = float(c.get("salience", 0.5))
            pen = float(c.get("uncertainty_penalty", 0.0))
            return (conf * 0.6 + sal * 0.4) - pen

        sorted_cands = sorted(candidates, key=score_cand, reverse=True)
        winner = sorted_cands[0]

        self.broadcast(
            sender_module=winner.get("module", "Arbiter"),
            salience_weight=score_cand(winner),
            content_summary=f"Elected winner from module '{winner.get('module')}' with score {score_cand(winner):.2f}",
            payload=winner
        )
        return winner
