"""
COLLISION Synaptic Working Memory & Global Workspace (GWT-SWM 2.0).

Implements Bernard Baars & Stanislas Dehaene's Global Workspace Theory with Hebbian Plasticity:
- Unconscious parallel cognitive modules compete for conscious blackboard access.
- Central Global Workspace broadcasts the winning hypothesis across all cognitive subsystems.
- Synaptic Working Memory tracks cognitive items with temporal decay, LTP potentiation,
  associative Hebbian co-activation links, and Episodic Replay consolidation.
"""

import time
import math
import heapq
from typing import Dict, Any, List, Optional, Tuple, Set

from collision.brain.schemas import (
    GlobalWorkspaceMessage,
    WorkingMemoryItem,
    HebbianAssociation,
    CognitiveModality
)


class SynapticWorkingMemory:
    """
    Dynamic Working Memory buffer simulating biological synaptic plasticity:
    - Items have salience (activation potential) that decays exponentially over time.
    - Repeated access strengthens salience (LTP - Long Term Potentiation).
    - Hebbian associative co-activation ("neurons that fire together wire together").
    - Capacity is bounded (Miller's Law 7±2 items in conscious focus).
    - Consolidation into persistent Long-Term Memory (LTM).
    """

    def __init__(self, capacity: int = 9, half_life_seconds: float = 300.0):
        self.capacity = capacity
        self.half_life_seconds = half_life_seconds
        self.items: Dict[str, WorkingMemoryItem] = {}
        self.recent_accessed_keys: List[str] = []

    def retain(
        self,
        key: str,
        value: Any,
        salience: float = 1.0,
        domain_tags: Optional[List[str]] = None
    ) -> WorkingMemoryItem:
        """Stores or reactivates an item in working memory and updates Hebbian links."""
        now = time.time()
        self._apply_decay(now)

        if key in self.items:
            item = self.items[key]
            item.value = value
            item.salience = min(2.5, item.salience + salience * 0.5)  # LTP reinforcement
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

        # Check for LTM Consolidation
        if item.access_count >= 3 and item.salience >= 1.4:
            item.is_consolidated = True

        # Form Hebbian co-activation links with recently accessed keys
        self._reinforce_hebbian_links(key, now)

        self._prune_if_needed()
        return item

    def recall(self, key: str, spread_activation: bool = True) -> Optional[Any]:
        """Retrieves an item, boosts its activation, and optionally spreads activation to associated keys."""
        now = time.time()
        self._apply_decay(now)

        if key in self.items:
            item = self.items[key]
            item.last_accessed = now
            item.access_count += 1
            item.salience = min(2.5, item.salience + 0.25)

            if item.access_count >= 3 and item.salience >= 1.4:
                item.is_consolidated = True

            self._reinforce_hebbian_links(key, now)

            # Spreading activation to associated keys
            if spread_activation and item.associations:
                for target_k, assoc in item.associations.items():
                    if target_k in self.items:
                        target_item = self.items[target_k]
                        target_item.salience = min(2.5, target_item.salience + assoc.synaptic_weight * 0.15)

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

    def _reinforce_hebbian_links(self, active_key: str, timestamp: float) -> None:
        """Strengthens synaptic association between active_key and recent keys."""
        if self.recent_accessed_keys:
            for prev_key in self.recent_accessed_keys[-3:]:
                if prev_key != active_key and prev_key in self.items:
                    # Link active_key -> prev_key
                    item = self.items[active_key]
                    if prev_key in item.associations:
                        assoc = item.associations[prev_key]
                        assoc.synaptic_weight = min(1.0, assoc.synaptic_weight + 0.1)
                        assoc.co_access_count += 1
                        assoc.last_reinforced = timestamp
                    else:
                        item.associations[prev_key] = HebbianAssociation(
                            target_key=prev_key,
                            synaptic_weight=0.5,
                            co_access_count=1,
                            last_reinforced=timestamp
                        )

        self.recent_accessed_keys.append(active_key)
        if len(self.recent_accessed_keys) > 10:
            self.recent_accessed_keys.pop(0)

    def _apply_decay(self, current_time: float) -> None:
        """Applies exponential decay to all items, protecting consolidated LTM items."""
        dead_keys = []
        for k, item in self.items.items():
            if item.is_consolidated:
                # Consolidated items have 10x slower decay and baseline floor
                effective_decay = item.decay_rate * 0.1
                dt = max(0.0, current_time - item.last_accessed)
                item.salience = max(0.8, item.salience * math.exp(-effective_decay * dt))
            else:
                dt = max(0.0, current_time - item.last_accessed)
                decayed = item.salience * math.exp(-item.decay_rate * dt)
                item.salience = decayed
                if decayed < 0.05:
                    dead_keys.append(k)

        for dk in dead_keys:
            del self.items[dk]

    def _prune_if_needed(self) -> None:
        """Retains only top-K salient items, prioritizing consolidated items."""
        if len(self.items) > self.capacity:
            sorted_items = sorted(
                self.items.items(),
                key=lambda x: (x[1].is_consolidated, x[1].salience),
                reverse=True
            )
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
                "domain_tags": v.domain_tags,
                "is_consolidated": v.is_consolidated,
                "associated_keys": list(v.associations.keys())
            }
            for k, v in self.items.items()
        }


class GlobalWorkspace:
    """
    Central conscious blackboard coordinating competing unconscious specialized modules:
    - Modules submit candidate proposals with salience scores.
    - The Workspace arbitrates and selects the most salient consensus.
    - Broadcasts winning state across all modules for coherent unified output.
    - Tracks Episodic Memory Replay for cross-query cognitive continuity.
    """

    def __init__(self):
        self.working_memory = SynapticWorkingMemory()
        self.broadcast_history: List[GlobalWorkspaceMessage] = []
        self.episodic_replay_buffer: List[Dict[str, Any]] = []
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

        # Retain in short-term synaptic working memory
        self.working_memory.retain(
            key=f"broadcast_{sender_module}_{self.cycle_counter}",
            value=content_summary,
            salience=salience_weight,
            domain_tags=[sender_module]
        )

        # Retain in episodic replay if salience is very high
        if salience_weight >= 0.85:
            self.episodic_replay_buffer.append({
                "cycle": self.cycle_counter,
                "sender": sender_module,
                "summary": content_summary,
                "payload": payload or {},
                "timestamp": msg.timestamp
            })
            if len(self.episodic_replay_buffer) > 20:
                self.episodic_replay_buffer.pop(0)

        return msg

    def arbitrate_candidates(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Selects the winning candidate from multiple competing specialized modules
        based on confidence, salience, and epistemic consistency.
        """
        if not candidates:
            return None

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
