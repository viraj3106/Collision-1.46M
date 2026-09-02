import os
import sys

# Resolve project root and insert into sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.tokenize import BPETokenizer

class CollisionTokenizer:
    def __init__(self, tokenizer_dir=None):
        if tokenizer_dir is None:
            tokenizer_dir = os.path.join(PROJECT_ROOT, "models", "collision-10m", "tokenizer")
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_dir)
        self.special_tokens = self.tokenizer.special_tokens

    def encode(self, text: str, bos: bool = True, eos: bool = False):
        return self.tokenizer.encode(text, bos=bos, eos=eos)

    def decode(self, token_ids) -> str:
        return self.tokenizer.decode(token_ids)
