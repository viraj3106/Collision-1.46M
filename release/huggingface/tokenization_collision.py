"""
COLLISION Tokenizer for Hugging Face Transformers Integration.
Allows AutoTokenizer.from_pretrained("collision-10M/Collision-1B", trust_remote_code=True).
"""

import os
import re
import json
from typing import List, Dict, Optional, Union

try:
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_base import BatchEncoding
except ImportError:
    class PreTrainedTokenizer:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class BatchEncoding(dict):
        pass


class CollisionTokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "vocab_file": "tokenizer/vocab.json",
        "merges_file": "tokenizer/merges.json",
    }
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        errors: str = "replace",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        pad_token: str = "[PAD]",
        model_max_length: int = 1024,
        **kwargs
    ):
        self.special_tokens_map_dict = {
            "[PAD]": 256,
            "[UNK]": 257,
            "[BOS]": 258,
            "[EOS]": 259,
        }
        self.inv_special_tokens = {v: k for k, v in self.special_tokens_map_dict.items()}

        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}

        # Default byte vocabulary (0-255)
        self.vocab = {bytes([i]): i for i in range(256)}
        for token_str, idx in self.special_tokens_map_dict.items():
            self.vocab[token_str.encode("utf-8")] = idx
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # Automatic fallback path resolution
        if not vocab_file or not os.path.exists(vocab_file):
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            for cand in [os.path.join(cur_dir, "tokenizer", "vocab.json"), os.path.join(cur_dir, "vocab.json")]:
                if os.path.exists(cand):
                    vocab_file = cand
                    break

        if not merges_file or not os.path.exists(merges_file):
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            for cand in [os.path.join(cur_dir, "tokenizer", "merges.json"), os.path.join(cur_dir, "merges.json")]:
                if os.path.exists(cand):
                    merges_file = cand
                    break

        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                json_vocab = json.load(f)
            self.vocab = {}
            for k, v in json_vocab.items():
                if k in self.special_tokens_map_dict:
                    self.vocab[k.encode("utf-8")] = v
                else:
                    try:
                        self.vocab[bytes.fromhex(k)] = v
                    except Exception:
                        self.vocab[k.encode("utf-8")] = v
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        if merges_file and os.path.exists(merges_file):
            with open(merges_file, "r", encoding="utf-8") as f:
                json_merges = json.load(f)
            self.merges = {}
            for k, v in json_merges.items():
                p0, p1 = map(int, k.split(","))
                self.merges[(p0, p1)] = v

        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            model_max_length=model_max_length,
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        return len(self.inverse_vocab)

    def get_vocab(self) -> Dict[str, int]:
        return {
            (k.decode("utf-8", errors="replace") if isinstance(k, bytes) else str(k)): v
            for k, v in self.vocab.items()
        }

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\s+|\w+|[^\w\s]", text)

    def _merge_word(self, word: List[int], pair: tuple, idx: int) -> List[int]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(idx)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word

    def encode_to_ids(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        if not text:
            return []
        chunks = re.findall(r"\s+|\w+|[^\w\s]", text)
        res = []
        if bos:
            res.append(self.special_tokens_map_dict["[BOS]"])

        for chunk in chunks:
            word = list(chunk.encode("utf-8"))
            for pair, merge_idx in self.merges.items():
                if len(word) <= 1:
                    break
                word = self._merge_word(word, pair, merge_idx)
            res.extend(word)

        if eos:
            res.append(self.special_tokens_map_dict["[EOS]"])
        return res

    def decode_from_ids(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        byte_chunks = []
        for i in ids:
            if i in self.inv_special_tokens:
                if not skip_special_tokens:
                    byte_chunks.append(self.inv_special_tokens[i].encode("utf-8"))
            elif i in self.inverse_vocab:
                val = self.inverse_vocab[i]
                byte_chunks.append(val if isinstance(val, bytes) else val.encode("utf-8"))
            else:
                if not skip_special_tokens:
                    byte_chunks.append(b"[UNK]")
        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        **kwargs
    ):
        if isinstance(text, str):
            ids = self.encode_to_ids(text, bos=True)
            if max_length and len(ids) > max_length:
                ids = ids[:max_length]
            mask = [1] * len(ids)

            if return_tensors == "pt":
                import torch
                return BatchEncoding({
                    "input_ids": torch.tensor([ids], dtype=torch.long),
                    "attention_mask": torch.tensor([mask], dtype=torch.long),
                })
            return BatchEncoding({"input_ids": ids, "attention_mask": mask})

        elif isinstance(text, list):
            all_ids = [self.encode_to_ids(t, bos=True) for t in text]
            if max_length:
                all_ids = [ids[:max_length] for ids in all_ids]
            max_l = max(len(ids) for ids in all_ids) if all_ids else 0

            padded_ids = []
            masks = []
            for ids in all_ids:
                pad_len = max_l - len(ids)
                padded_ids.append(ids + [self.special_tokens_map_dict["[PAD]"]] * pad_len)
                masks.append([1] * len(ids) + [0] * pad_len)

            if return_tensors == "pt":
                import torch
                return BatchEncoding({
                    "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(masks, dtype=torch.long),
                })
            return BatchEncoding({"input_ids": padded_ids, "attention_mask": masks})

    def decode(self, token_ids: Union[List[int], "torch.Tensor"], skip_special_tokens: bool = True, **kwargs) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self.decode_from_ids(token_ids, skip_special_tokens=skip_special_tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, "vocab.json")
        merges_file = os.path.join(save_directory, "merges.json")
        json_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        json_vocab = {k.hex() if isinstance(k, bytes) else str(k): v for k, v in self.vocab.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(json_vocab, f, indent=2)
        with open(merges_file, "w", encoding="utf-8") as f:
            json.dump(json_merges, f, indent=2)
        return (vocab_file, merges_file)
