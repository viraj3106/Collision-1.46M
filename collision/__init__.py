# COLLISION package definition

class CollisionError(Exception):
    """Base exception class for all COLLISION errors."""
    pass

from collision.nlp import CollisionNLPEngine, CollisionNLPProcessor

__all__ = ["CollisionError", "CollisionNLPEngine", "CollisionNLPProcessor"]
