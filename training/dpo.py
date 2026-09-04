import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_sequence_logprobs(model, input_ids, prompt_lens, pad_token_id=0):
    """
    Computes sum of log probabilities for response tokens in input_ids.
    input_ids: tensor of shape (batch_size, seq_len)
    prompt_lens: list or tensor of prompt lengths per sequence
    pad_token_id: token ID for padding
    """
    batch_size, seq_len = input_ids.shape
    if seq_len < 2:
        return torch.zeros(batch_size, device=input_ids.device)

    # Causal shifting: logits for input_ids[:, :-1], targets are input_ids[:, 1:]
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]

    logits, _ = model(x)
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log-probabilities of target tokens
    per_token_logps = torch.gather(log_probs, 2, y.unsqueeze(-1)).squeeze(-1)

    # Mask: target index t (0 to seq_len-2) corresponds to target token input_ids[:, t+1]
    mask = torch.zeros_like(per_token_logps, dtype=torch.float32)
    for b in range(batch_size):
        p_len = prompt_lens[b] if isinstance(prompt_lens, (list, tuple)) else prompt_lens[b].item()
        for t in range(seq_len - 1):
            target_idx = t + 1
            if target_idx >= p_len and y[b, t].item() != pad_token_id:
                mask[b, t] = 1.0

    seq_logprobs = (per_token_logps * mask).sum(dim=-1)
    return seq_logprobs

def canonical_dpo_loss(policy_model, reference_model, chosen_ids, rejected_ids, chosen_prompt_lens, rejected_prompt_lens, beta=0.1, pad_token_id=0):
    """
    Computes standard Canonical DPO loss between policy_model and frozen reference_model.
    """
    # Policy model log-probabilities
    chosen_logp_policy = compute_sequence_logprobs(policy_model, chosen_ids, chosen_prompt_lens, pad_token_id)
    rejected_logp_policy = compute_sequence_logprobs(policy_model, rejected_ids, rejected_prompt_lens, pad_token_id)

    # Frozen Reference model log-probabilities
    with torch.no_grad():
        chosen_logp_reference = compute_sequence_logprobs(reference_model, chosen_ids, chosen_prompt_lens, pad_token_id)
        rejected_logp_reference = compute_sequence_logprobs(reference_model, rejected_ids, rejected_prompt_lens, pad_token_id)

    policy_logratio = chosen_logp_policy - rejected_logp_policy
    reference_logratio = chosen_logp_reference - rejected_logp_reference

    logits = beta * (policy_logratio - reference_logratio)
    loss = -F.logsigmoid(logits).mean()

    return loss, chosen_logp_policy, rejected_logp_policy, chosen_logp_reference, rejected_logp_reference
