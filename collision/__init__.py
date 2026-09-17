# COLLISION package definition

class CollisionError(Exception):
    """Base exception class for all COLLISION errors."""
    pass

from collision.nlp import CollisionNLPEngine, CollisionNLPProcessor
from collision.brain import CollisionBrain, SynapticCognitiveBrain, get_collision_brain

__all__ = [
    "CollisionError",
    "CollisionNLPEngine",
    "CollisionNLPProcessor",
    "CollisionBrain",
    "SynapticCognitiveBrain",
    "get_collision_brain"
]
