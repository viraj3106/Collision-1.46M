"""
COLLISION Synaptic-GWT Cognitive Brain — Core Data Schemas.

Defines formal models for:
- Global Workspace state & Hebbian message broadcasting
- Graph-of-Thoughts (GoT 2.0) multi-paradigm DAGs & Hegelian Dialectics
- Dual-Process System 1/2 routing & Multi-Scale Epistemic Uncertainty
- NLP Information Density, Renyi/Shannon Entropy, & Semantic Knowledge Triples
- Brain Metacognitive Trace, Bias/Fallacy Audits & Visual Graph Rendering
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
    AXIOM = "AXIOM"
    HYPOTHESIS = "HYPOTHESIS"
    COUNTERFACTUAL_SIMULATION = "COUNTERFACTUAL_SIMULATION"
    CAUSAL_INFERENCE = "CAUSAL_INFERENCE"
    BOUNDARY_CONSTRAINT = "BOUNDARY_CONSTRAINT"
    ACTIONABLE_HEURISTIC = "ACTIONABLE_HEURISTIC"


class CognitiveModality(str, Enum):
    SYSTEM_1_REFLEX = "SYSTEM_1_REFLEX"
    SYSTEM_2_DELIBERATION = "SYSTEM_2_DELIBERATION"
    DIALECTICAL_SYNTHESIS = "DIALECTICAL_SYNTHESIS"
    CROSS_DOMAIN_SYNAPSE = "CROSS_DOMAIN_SYNAPSE"
    NEURO_SYMBOLIC_VERIFICATION = "NEURO_SYMBOLIC_VERIFICATION"


class BiasSeverity(str, Enum):
    ADVISORY = "ADVISORY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class CognitiveBiasAudit(BaseModel):
    """Structured record of detected cognitive bias or formal logical fallacy."""
    bias_type: str
    severity: BiasSeverity = BiasSeverity.WARNING
    detected_pattern: str
    detected_snippet: str
    explanation: str
    remediation_suggestion: str
    confidence_penalty: float = 0.05


class SemanticTriple(BaseModel):
    """Structured knowledge triplet: (Subject, Predicate, Object, Context)."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    context: Optional[str] = None
    polarity: str = "AFFIRMATIVE"


class LinguisticComplexityMetrics(BaseModel):
    """Multi-scale linguistic and structural complexity indicators."""
    renyi_entropy_order2: float = 0.0
    flesch_reading_ease: float = 65.0
    gunning_fog_estimate: float = 8.0
    syntactic_depth_estimate: int = 3
    token_surprisal_avg: float = 2.5


class InformationEntropyProfile(BaseModel):
    """Information-theoretic density and complexity metrics across input tokens."""
    total_tokens: int
    unique_tokens: int
    shannon_entropy: float
    information_density: float  # Bits per token
    lexical_diversity: float   # Type-Token Ratio (TTR)
    salience_spikes: List[str] = Field(default_factory=list)
    epistemic_ambiguity_score: float = 0.0
    renyi_entropy: float = 0.0
    complexity: Optional[LinguisticComplexityMetrics] = None


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
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DialecticalGraph(BaseModel):
    """DAG of thoughts representing Thesis, Antithesis, and Synthesis convergence."""
    nodes: Dict[str, ThoughtNode] = Field(default_factory=dict)
    edges: List[Tuple[str, str]] = Field(default_factory=list)  # (source_id, target_id)
    thesis_id: Optional[str] = None
    antithesis_id: Optional[str] = None
    synthesis_id: Optional[str] = None
    dialectic_resolved: bool = True
    coherence_score: float = 1.0
    adversarial_resilience_score: float = 0.95

    def add_node(self, node: ThoughtNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, source_id: str, target_id: str) -> None:
        if source_id in self.nodes and target_id in self.nodes:
            edge = (source_id, target_id)
            if edge not in self.edges:
                self.edges.append(edge)
            if source_id not in self.nodes[target_id].dependencies:
                self.nodes[target_id].dependencies.append(source_id)

    def topological_sort(self) -> List[str]:
        """Returns node IDs in topological execution order."""
        in_degree = {nid: 0 for nid in self.nodes}
        for src, dst in self.edges:
            if dst in in_degree:
                in_degree[dst] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for src, dst in self.edges:
                if src == curr:
                    in_degree[dst] -= 1
                    if in_degree[dst] == 0:
                        queue.append(dst)

        # Fallback for any unvisited nodes (e.g. disconnected)
        for nid in self.nodes:
            if nid not in order:
                order.append(nid)
        return order

    def to_mermaid(self) -> str:
        """Generates Mermaid flowchart diagram representing this Graph of Thoughts."""
        lines = ["flowchart TD"]
        # Styling classes
        lines.append("    classDef thesis fill:#e8f0fe,stroke:#4285f4,stroke-width:2px,color:#1a73e8;")
        lines.append("    classDef antithesis fill:#fce8e6,stroke:#ea4335,stroke-width:2px,color:#d93025;")
        lines.append("    classDef synthesis fill:#e6f4ea,stroke:#34a853,stroke-width:2px,color:#137333;")
        lines.append("    classDef deduction fill:#fef7e0,stroke:#fbbc04,stroke-width:1.5px,color:#b06000;")
        lines.append("    classDef general fill:#f1f3f4,stroke:#5f6368,stroke-width:1px,color:#3c4043;")

        for nid, node in self.nodes.items():
            clean_content = node.content.replace('"', "'").replace("\n", " ")
            if len(clean_content) > 55:
                clean_content = clean_content[:52] + "..."
            node_label = f"{node.thought_type.value}: {clean_content}"
            lines.append(f'    {nid}["{node_label}"]')

        for src, dst in self.edges:
            lines.append(f"    {src} --> {dst}")

        for nid, node in self.nodes.items():
            if node.thought_type == ThoughtType.THESIS:
                lines.append(f"    class {nid} thesis")
            elif node.thought_type in (ThoughtType.ANTITHESIS, ThoughtType.REFUTATION):
                lines.append(f"    class {nid} antithesis")
            elif node.thought_type == ThoughtType.SYNTHESIS:
                lines.append(f"    class {nid} synthesis")
            elif node.thought_type in (ThoughtType.DEDUCTION, ThoughtType.CAUSAL_INFERENCE):
                lines.append(f"    class {nid} deduction")
            else:
                lines.append(f"    class {nid} general")

        return "\n".join(lines)

    def to_ascii_tree(self) -> str:
        """Renders an ASCII text tree of the reasoning graph."""
        lines = ["[Graph of Thoughts DAG]"]
        ordered = self.topological_sort()
        for idx, nid in enumerate(ordered):
            node = self.nodes[nid]
            prefix = "\\-- " if idx == len(ordered) - 1 else "|-- "
            dep_str = f" (depends on: {', '.join(node.dependencies)})" if node.dependencies else ""
            lines.append(f"{prefix}[{node.thought_type.value}] ({nid}) - {node.content[:60]}...{dep_str}")
        return "\n".join(lines)


class GlobalWorkspaceMessage(BaseModel):
    """Message broadcast onto the central conscious blackboard."""
    sender_module: str
    salience_weight: float
    content_summary: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    broadcast_cycle: int = 1
    timestamp: float


class HebbianAssociation(BaseModel):
    """Synaptic association linking two working memory items."""
    target_key: str
    synaptic_weight: float = 0.5
    co_access_count: int = 1
    last_reinforced: float = 0.0


class WorkingMemoryItem(BaseModel):
    """Item held in short-term synaptic working memory."""
    key: str
    value: Any
    salience: float = 1.0
    decay_rate: float = 0.05
    access_count: int = 1
    last_accessed: float
    domain_tags: List[str] = Field(default_factory=list)
    associations: Dict[str, HebbianAssociation] = Field(default_factory=dict)
    is_consolidated: bool = False


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
    bias_audits: List[CognitiveBiasAudit] = Field(default_factory=list)
    stage_latencies: Dict[str, float] = Field(default_factory=dict)
    mermaid_diagram: Optional[str] = None
    ascii_tree: Optional[str] = None
    total_brain_latency_ms: float = 0.0


class BrainResponse(BaseModel):
    """Unified response produced by the COLLISION Synaptic Brain."""
    query: str
    answer: str
    modality: CognitiveModality
    confidence: float = 1.0
    epistemic_certainty: float = 1.0
    axiomatic_confidence: float = 0.95
    adversarial_resilience: float = 0.92
    merit_score: float = 0.96
    dialectical_resolution: Optional[str] = None
    primary_domain: str = "General"
    triples: List[SemanticTriple] = Field(default_factory=list)
    key_insights: List[str] = Field(default_factory=list)
    followup_hypotheses: List[str] = Field(default_factory=list)
    bias_audits: List[CognitiveBiasAudit] = Field(default_factory=list)
    graph_mermaid: Optional[str] = None
    graph_ascii: Optional[str] = None
    trace: Optional[BrainCognitiveTrace] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
