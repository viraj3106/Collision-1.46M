"""
COLLISION Synaptic-GWT Cognitive Brain — Core Data Schemas.

Defines formal models for:
- Global Workspace state & message broadcasting
- Graph-of-Thoughts (GoT) nodes & Hegelian Dialectical DAGs
- Dual-Process System 1/2 routing & Epistemic Uncertainty
- NLP Information Density Vectors & Semantic Knowledge Triples
- Brain Metacognitive Trace & Audit Records
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field


class ThoughtType(str, Enum):
    THESIS = "THESIS"
    ANTITHESIS = "ANTITHESIS"
    SYNTHESIS = "SYNTHESIS"
    DEDUCTION = "DEDUCTION"
    REFUTATION = "REFUTATION"
    META_CRITIQUE = "META_CRITIQUE"
    EMPIRICAL_OBSERVATION = "EMPIRICAL_OBSERVATION"


class CognitiveModality(str, Enum):
    SYSTEM_1_REFLEX = "SYSTEM_1_REFLEX"
    SYSTEM_2_DELIBERATION = "SYSTEM_2_DELIBERATION"
    DIALECTICAL_SYNTHESIS = "DIALECTICAL_SYNTHESIS"
    CROSS_DOMAIN_SYNAPSE = "CROSS_DOMAIN_SYNAPSE"
    NEURO_SYMBOLIC_VERIFICATION = "NEURO_SYMBOLIC_VERIFICATION"


class SemanticTriple(BaseModel):
    """Structured knowledge triplet: (Subject, Predicate, Object, Context)."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    context: Optional[str] = None
    polarity: str = "AFFIRMATIVE"


class InformationEntropyProfile(BaseModel):
    """Information-theoretic density and complexity metrics across input tokens."""
    total_tokens: int
    unique_tokens: int
    shannon_entropy: float
    information_density: float  # Bits per token
    lexical_diversity: float   # Type-Token Ratio (TTR)
    salience_spikes: List[str] = Field(default_factory=list)
    epistemic_ambiguity_score: float = 0.0


class ThoughtNode(BaseModel):
    """Single node in the non-linear Graph of Thoughts."""
    node_id: str
    thought_type: ThoughtType
    content: str
    confidence: float = 1.0
    dependencies: List[str] = Field(default_factory=list)
    evidence_sources: List[str] = Field(default_factory=list)
    entropy_score: float = 0.0
    validation_status: str = "VALIDATED"  # VALIDATED, CONTESTED, RECONCILED, REFUTED
    rationale: Optional[str] = None


class DialecticalGraph(BaseModel):
    """DAG of thoughts representing Thesis, Antithesis, and Synthesis convergence."""
    nodes: Dict[str, ThoughtNode] = Field(default_factory=dict)
    edges: List[Tuple[str, str]] = Field(default_factory=list)  # (source_id, target_id)
    thesis_id: Optional[str] = None
    antithesis_id: Optional[str] = None
    synthesis_id: Optional[str] = None
    dialectic_resolved: bool = True
    coherence_score: float = 1.0

    def add_node(self, node: ThoughtNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, source_id: str, target_id: str) -> None:
        if source_id in self.nodes and target_id in self.nodes:
            self.edges.append((source_id, target_id))
            if source_id not in self.nodes[target_id].dependencies:
                self.nodes[target_id].dependencies.append(source_id)


class GlobalWorkspaceMessage(BaseModel):
    """Message broadcast onto the central conscious blackboard."""
    sender_module: str
    salience_weight: float
    content_summary: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    broadcast_cycle: int = 1
    timestamp: float


class WorkingMemoryItem(BaseModel):
    """Item held in short-term synaptic working memory."""
    key: str
    value: Any
    salience: float = 1.0
    decay_rate: float = 0.05
    access_count: int = 1
    last_accessed: float
    domain_tags: List[str] = Field(default_factory=list)


class BrainCognitiveTrace(BaseModel):
    """Explainable introspective execution trace of the brain."""
    modality: CognitiveModality
    system_1_latency_ms: float = 0.0
    system_2_latency_ms: float = 0.0
    epistemic_entropy: float = 0.0
    activated_modules: List[str] = Field(default_factory=list)
    broadcast_messages: List[GlobalWorkspaceMessage] = Field(default_factory=list)
    graph_of_thoughts: Optional[DialecticalGraph] = None
    extracted_triples: List[SemanticTriple] = Field(default_factory=list)
    bias_check_notes: List[str] = Field(default_factory=list)
    total_brain_latency_ms: float = 0.0


class BrainResponse(BaseModel):
    """Unified response produced by the COLLISION Synaptic Brain."""
    query: str
    answer: str
    modality: CognitiveModality
    confidence: float = 1.0
    epistemic_certainty: float = 1.0
    dialectical_resolution: Optional[str] = None
    primary_domain: str = "General"
    triples: List[SemanticTriple] = Field(default_factory=list)
    key_insights: List[str] = Field(default_factory=list)
    followup_hypotheses: List[str] = Field(default_factory=list)
    trace: Optional[BrainCognitiveTrace] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
