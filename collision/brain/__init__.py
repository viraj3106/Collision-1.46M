"""
COLLISION Synaptic-GWT Cognitive Brain Subsystem.

Provides:
- Global Workspace Theory & Synaptic Working Memory (`GlobalWorkspace`, `SynapticWorkingMemory`)
- Dual-Process System 1/2 Controller (`DualProcessArbiter`, `EpistemicUncertaintyQuantifier`, `MetacognitiveCritic`)
- Graph-of-Thoughts Hegelian Dialectic Engine (`HegelianDialecticEngine`, `GraphOfThoughtReasoner`, `DialecticalGraph`)
- Synaptic NLP Information-Theoretic Engine (`SemanticTripletExtractor`, `ContextualPolysemyDisambiguator`)
- Cross-Domain Knowledge Lattice (`CrossDomainKnowledgeLattice`)
- Flagship Brain Engine (`CollisionBrain`, `SynapticCognitiveBrain`, `get_collision_brain`)
"""

from collision.brain.schemas import (
    ThoughtType,
    ThoughtNode,
    DialecticalGraph,
    CognitiveModality,
    SemanticTriple,
    InformationEntropyProfile,
    GlobalWorkspaceMessage,
    WorkingMemoryItem,
    BrainCognitiveTrace,
    BrainResponse
)
from collision.brain.workspace import GlobalWorkspace, SynapticWorkingMemory
from collision.brain.dual_process import (
    DualProcessArbiter,
    EpistemicUncertaintyQuantifier,
    MetacognitiveCritic
)
from collision.brain.graph_of_thought import (
    HegelianDialecticEngine,
    GraphOfThoughtReasoner
)
from collision.brain.synaptic_nlp import (
    SemanticTripletExtractor,
    ContextualPolysemyDisambiguator
)
from collision.brain.knowledge_lattice import CrossDomainKnowledgeLattice
from collision.brain.engine import (
    CollisionBrain,
    SynapticCognitiveBrain,
    get_collision_brain
)

__all__ = [
    "CollisionBrain",
    "SynapticCognitiveBrain",
    "get_collision_brain",
    "GlobalWorkspace",
    "SynapticWorkingMemory",
    "DualProcessArbiter",
    "EpistemicUncertaintyQuantifier",
    "MetacognitiveCritic",
    "HegelianDialecticEngine",
    "GraphOfThoughtReasoner",
    "DialecticalGraph",
    "ThoughtNode",
    "ThoughtType",
    "CognitiveModality",
    "SemanticTriple",
    "InformationEntropyProfile",
    "GlobalWorkspaceMessage",
    "WorkingMemoryItem",
    "BrainCognitiveTrace",
    "BrainResponse",
    "SemanticTripletExtractor",
    "ContextualPolysemyDisambiguator",
    "CrossDomainKnowledgeLattice"
]
