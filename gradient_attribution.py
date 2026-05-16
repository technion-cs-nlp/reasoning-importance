from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc
from torch.utils.checkpoint import checkpoint


@torch.enable_grad
def calculate_output_to_output_gradient_matrix(
    model,
    tokenizer,
    prompt: str,
    cot_start_idx: int,
    sentence_token_borders: Optional[List[Tuple[int, int]]],
    use_checkpointing: bool = True,
    are_token_borders_absolute: bool = False,
    use_gradient_times_input: bool = False,
    verbose: bool = False,
) -> torch.tensor:
    """
    Gradient attribution method that calculates gradient matrix I where I[i,j] represents
    the gradient of the sum of all token probabilities that belong to step j w.r.t the embeddings
    of all tokens in step j.

    Args:
        model: Pre-trained language model
        tokenizer: Corresponding tokenizer
        prompt: Full input+output string
        cot_start_idx: Index where the CoT begins in the tokenized prompt
        sentence_token_borders: Optional list of (start, end) token indices for sentences in the output.
        use_checkpointing: Whether to use gradient checkpointing to save memory
        are_token_borders_absolute:
            Whether the provided token borders are absolute indices in the full prompt or relative to the CoT start
        use_gradient_times_input:
            Whether to use norm(gradient times input) as the attribution score instead of just norm(gradient).
        verbose: Whether to show progress bar and additional info
    Returns:
        torch.tensor: Gradient matrix of shape (output_length, input_length)
    """
    # Clear cache and force garbage collection
    torch.cuda.empty_cache()
    gc.collect()

    if use_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # Enable gradient checkpointing for the model
        model.gradient_checkpointing_enable()

    # Tokenize the full prompt
    token_ids = tokenizer.encode(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    output_length = len(token_ids[0]) - cot_start_idx - 1

    # Get embeddings with gradient tracking
    embeddings = model.get_input_embeddings()(token_ids)
    input_embeds, output_embeds = (
        embeddings[:, :-output_length, :],
        embeddings[:, -output_length:, :],
    )
    embeddings = torch.cat((input_embeds, output_embeds), dim=1)
    embeddings = embeddings.contiguous().to(model.device).to(model.dtype)
    embeddings.requires_grad_(True)

    if use_checkpointing:
        # Use gradient checkpointing for forward pass
        def checkpointed_forward(embeddings):
            return model(inputs_embeds=embeddings).logits

        # Forward pass with checkpointing to save memory
        logits = checkpoint(checkpointed_forward, embeddings, use_reentrant=False)[0]
    else:
        outputs = model(inputs_embeds=embeddings)
        logits = outputs.logits[0]  # Shape: (seq_len, vocab_size)

    # Calculate attribution in a sentence-level
    influence_matrix = torch.zeros(
        (len(sentence_token_borders), len(sentence_token_borders)),
        device="cpu",
    )
    iter = sentence_token_borders if not verbose else tqdm(sentence_token_borders)
    for k, (sent_start_k, sent_end_k) in enumerate(iter):
        offset = 0 if are_token_borders_absolute else cot_start_idx
        output_positions = torch.tensor(
            range(
                offset + sent_start_k,
                offset + sent_end_k,
            )
        )
        target_tokens = token_ids[0, output_positions]
        try:
            probs = F.softmax(
                logits[output_positions - 1], dim=-1
            )  # Logits predict target tokens, offset by 1
            target_probs = probs.gather(
                dim=-1, index=target_tokens.unsqueeze(-1)
            ).squeeze()
        except Exception as e:
            print(
                f"Error in softmax computation in sentence idx={k}/{len(iter)-1}, with output positions {output_positions} (out of total {output_length}) with full prompt: {prompt}"
            )
            raise e

        total_target_prob = target_probs.sum()
        grads = torch.autograd.grad(
            outputs=total_target_prob,
            inputs=output_embeds,  # Only consider output embeddings
            grad_outputs=torch.ones_like(total_target_prob),
            retain_graph=k < len(sentence_token_borders) - 1,
            create_graph=False,
        )[0]

        for j, (src_sent_start, src_sent_end) in enumerate(sentence_token_borders):
            sent_grads = grads[0, src_sent_start:src_sent_end]
            if use_gradient_times_input:
                sent_embeds = output_embeds[0, src_sent_start:src_sent_end]
                influence_matrix[k, j] = (sent_grads * sent_embeds).norm().detach()
            else:
                influence_matrix[k, j] = sent_grads.norm().detach()

        del grads, probs
        torch.cuda.empty_cache()
        gc.collect()

    return influence_matrix.to(torch.float32)
