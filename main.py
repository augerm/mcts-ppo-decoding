from mistral.cache import RotatingBufferCache
import logging
import torch
import fire
from typing import List
from pathlib import Path

from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from tree_search_decoding.mcts import mcts
from tree_search_decoding.game_state import GameState
from tree_search_decoding.policy_model import PolicyModel
from tree_search_decoding.value_model import ValueModel


def sample_top_p(probs: torch.Tensor, p: float):
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)

# def find_best_token(logits: torch.Tensor, temperature: float, top_p: float):
#     game_state
#     pass

@torch.inference_mode()
def generate(prompts: List[str], model: Transformer, tokenizer: Tokenizer, *, max_tokens: int,  temperature: float, chunk_size: int = None, use_mcts_decoding: bool = False):
    model = model.eval()
    value_model = ValueModel()
    policy_model = PolicyModel()
    B, V = len(prompts), model.args.vocab_size

    # Tokenize
    encoded_prompts = [tokenizer.encode(prompt, bos=True) for prompt in prompts]
    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    cache_window = max(seqlens) + max_tokens
    if model.args.sliding_window is not None and cache_window > model.args.sliding_window:
        cache_window = model.args.sliding_window
    cache = RotatingBufferCache(
        model.n_local_layers,
        model.args.max_batch_size,
        cache_window,
        model.args.n_kv_heads,
        model.args.head_dim,
    )
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()
    
    # Bookkeeping
    logprobs = [[] for _ in range(B)]
    last_token_prelogits = None

    # One chunk if size not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len

    # Encode prompt by chunks
    for s in range(0, max_prompt_len, chunk_size):
        prompt_chunks = [p[s:s+chunk_size] for p in encoded_prompts]
        assert all(len(p) > 0 for p in prompt_chunks)
        prelogits = model.forward(
            torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
            seqlens=[len(p) for p in prompt_chunks],
            cache=cache
        )
        logits = torch.log_softmax(prelogits, dim=-1)

        if last_token_prelogits is not None:
            # Pass > 1
            last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
            for i_seq in range(B):
                logprobs[i_seq].append(last_token_logits[i_seq, prompt_chunks[i_seq][0]].item())

        offset = 0
        for i_seq, sequence in enumerate(prompt_chunks):
            logprobs[i_seq].extend([logits[offset + i, sequence[i + 1]].item() for i in range(len(sequence) - 1)])
            offset += len(sequence)

        last_token_prelogits = prelogits.index_select(0, torch.tensor([len(p) for p in prompt_chunks], device=prelogits.device).cumsum(dim=0) - 1)
        assert last_token_prelogits.shape == (B, V)

    # decode
    generated_tokens = []
    assert last_token_prelogits is not None
    detailed_logprobs = [[] for _ in range(B)]  # For detailed logging
    for i_token in range(max_tokens):
        next_token = None
        if use_mcts_decoding:
            game_state = GameState(model, policy_model, value_model, tokenizer, generated_tokens)
            next_token = mcts(game_state, iterations = 1)
        else:
            next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)

        last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
        for i in range(B):
            logprobs[i].append(last_token_logits[i, next_token[i]].item())
            all_token_probs = torch.softmax(last_token_prelogits[i], dim=-1)
            top_tokens = torch.topk(all_token_probs, k=10, dim=-1)
            
            decoded_tokens = [tokenizer.decode([token_id]) for token_id in top_tokens.indices.tolist()]
            token_probs = top_tokens.values.tolist()
            
            # Store decoded tokens and their probabilities
            detailed_log_entry = list(zip(decoded_tokens, token_probs))
            detailed_logprobs[i].append(detailed_log_entry)

        generated_tokens.append(next_token[:, None])
        last_token_prelogits = model.forward(next_token, seqlens=[1] * len(prompts), cache=cache)
        assert last_token_prelogits.shape == (B, V)

    generated_words = []
    if generated_tokens:
        generated_tokens = torch.cat(generated_tokens, 1)
        for i, x in enumerate(encoded_prompts):
            generated_words.append(tokenizer.decode(x + generated_tokens[i].tolist()))

    return generated_words, logprobs, detailed_logprobs


def interactive(model_path: str, max_tokens: int = 35, temperature: float = 0.7, instruct: bool = False, use_mcts_decoding: bool = False):
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)

    while True:
        prompt = input("Prompt: ")
        if instruct:
            prompt = f"[INST] {prompt} [/INST]"
        res, _logprobs = generate(
            [prompt],
            transformer,
            tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
            use_mcts_decoding=use_mcts_decoding,
        )
        print(res[0])
        print("=====================")


def demo(
    model_path: str, max_tokens: int = 35, temperature: float = 0, num_pipeline_ranks=1, use_mcts_decoding: bool = False
):
    if num_pipeline_ranks > 1:
        torch.distributed.init_process_group()
        torch.cuda.set_device(torch.distributed.get_rank())
        should_print = torch.distributed.get_rank() == 0
    else:
        should_print = True
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(
        Path(model_path), max_batch_size=3, num_pipeline_ranks=num_pipeline_ranks
    )

    res, _logprobs, detailed_logprobs = generate(
        [
            "Question: Osvaldo lies. Phoebe says Osvaldo lies. Kandi says Phoebe tells the truth. Crista says Kandi tells the truth. Delbert says Crista lies. Does Delbert tell the truth?\nAnswer: ",
        ],
        transformer,
        tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
        use_mcts_decoding=use_mcts_decoding,
    )
    newline = '\n'
    print("Detailed Log Probs:")

    for sublist in detailed_logprobs:
        for tup in sublist:
            # Creating a string representation of each tuple
            tup_str = ', '.join([str(elem) for elem in tup])
            print(f"{newline}{tup_str}")
        print(newline)  # Extra newline for separating between sublists

    if should_print:
        for x,l in zip(res, _logprobs):
            print(x)
            logging.debug('Logprobs: %s',l)
            print("=====================")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire({
        "interactive": interactive,
        "demo": demo,
    })
