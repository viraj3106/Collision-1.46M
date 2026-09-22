"""
COLLISION Fast Parallel Diffusion & Block-Jacobi Sampler (Pillar 1).

Generates complete sentences, code, and answers in 2-6 parallel refinement iterations
instead of hundreds of sequential autoregressive decoding steps.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from model.parallel_decoder import CollisionParallelDecoder


class ParallelSampler:
    """
    Confidence-Guided Iterative Masked Sampler for COLLISION Parallel Decoder.
    Decodes an entire sequence in parallel using confidence-aware unmasking.
    """
    def __init__(self, model: CollisionParallelDecoder, mask_token_id: int = 3):
        self.model = model
        self.mask_token_id = mask_token_id
        self.model.eval()

    @torch.no_grad()
    def sample(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        num_iterations: int = 4,
        temperature: float = 0.7,
        top_k: int = 40,
    ) -> Dict[str, Any]:
        """
        Parallel diffusion generation.
        Args:
            prompt_ids: (1, P) Tensor of prompt token ids
            max_new_tokens: Length of the generation block
            num_iterations: Number of parallel refinement iterations (e.g. 4)
            temperature: Sampling temperature
            top_k: Top-k filtering for token sampling
        Returns:
            Dictionary containing generated_ids, token_count, iterations, and step-by-step history.
        """
        device = prompt_ids.device
        P = prompt_ids.shape[1]
        total_len = P + max_new_tokens

        # 1. Initialize output buffer with prompt + [MASK] tokens
        output_ids = torch.full((1, total_len), self.mask_token_id, dtype=torch.long, device=device)
        output_ids[0, :P] = prompt_ids[0]

        history = [output_ids.clone()]

        # Linear unmasking schedule
        for step in range(num_iterations):
            step_tensor = torch.tensor([step], device=device)

            # Forward pass: Predict all tokens and their confidence scores in parallel
            logits, confidence = self.model(output_ids, step=step_tensor)

            # Restrict sampling to generated region
            gen_logits = logits[0, P:] / max(temperature, 1e-4)
            # Never sample the mask token itself
            gen_logits[:, self.mask_token_id] = -float('Inf')
            gen_confidence = confidence[0, P:]

            # Apply Top-K filtering
            if top_k > 0:
                v, _ = torch.topk(gen_logits, min(top_k, gen_logits.size(-1)))
                gen_logits[gen_logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(gen_logits, dim=-1)
            predicted_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Calculate how many tokens should remain masked at this step
            # Step 0: unmask 25%, Step 1: 50%, Step 2: 75%, Step 3: 100%
            unmask_ratio = (step + 1) / num_iterations
            num_to_unmask = int(unmask_ratio * max_new_tokens)

            # Rank positions by confidence score
            _, high_conf_indices = torch.topk(gen_confidence, k=min(num_to_unmask, max_new_tokens))

            # Update high confidence tokens
            updated_gen_ids = output_ids[0, P:].clone()
            updated_gen_ids[high_conf_indices] = predicted_tokens[high_conf_indices]

            # In the final step, fill all remaining masks
            if step == num_iterations - 1:
                updated_gen_ids = predicted_tokens

            output_ids[0, P:] = updated_gen_ids
            history.append(output_ids.clone())

        return {
            "generated_ids": output_ids,
            "prompt_len": P,
            "new_tokens": max_new_tokens,
            "iterations_used": num_iterations,
            "history": history
        }
