import sys
import os
import torch
from types import SimpleNamespace

# Ensure src is on sys.path so we can import modules written under src/
sys.path.insert(0, os.path.abspath("src"))

from generate import BasicGenerator


class DummyTokenizer:
    def __init__(self):
        self.eos_token = "</s>"
        self.pad_token = None

    def encode(self, text, return_tensors=None):
        # represent prompt as a single token id (length=1)
        return torch.tensor([[1]])

    def decode(self, ids):
        # ids may be a tensor of ids; return a deterministic string
        return "<GEN>"

    def convert_ids_to_tokens(self, ids):
        # return a list of token strings; first token starts with a space
        length = ids.shape[0]
        return [" a" if i == 0 else f"tok{i}" for i in range(length)]

    def tokenize(self, s):
        return [" "]


class DummyModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, return_dict_in_generate=False, output_scores=False):
        input_len = input_ids.shape[1]
        gen_len = max_new_tokens
        # sequences: shape (1, input_len + gen_len)
        seq = torch.arange(0, input_len + gen_len).unsqueeze(0)

        # create dummy scores only if requested
        scores = None
        if output_scores:
            vocab_size = 8
            scores = [torch.randn(1, vocab_size) for _ in range(gen_len)]

        return SimpleNamespace(sequences=seq, scores=scores)

    def compute_transition_scores(self, sequences, scores, normalize_logits=True):
        # compute a fake logprob per generated token by taking the max logits
        # scores is a list of tensors shape (1, vocab_size)
        vals = [torch.tensor(0.0) for _ in range(len(scores))]
        return [torch.stack(vals)]

    def __call__(self, generated_tokens, output_attentions=False):
        # Return dummy attentions: list where last element has shape
        # (batch=1, heads=2, seq_len, seq_len)
        gen_len = generated_tokens.shape[1]
        att = torch.rand(1, 2, gen_len, gen_len)
        return SimpleNamespace(attentions=[att])


def test_basic_generator_generate_and_generate_attn():
    # Create instance without running the real constructor (which would download models)
    gen = BasicGenerator.__new__(BasicGenerator)
    gen.tokenizer = DummyTokenizer()
    gen.model = DummyModel()
    gen.model_config = SimpleNamespace(model_type="other")
    gen.space_token = gen.tokenizer.tokenize(" ")[0]

    # Test generate with logprobs
    text, tokens, logprobs = gen.generate("hello", max_length=3, return_logprobs=True)
    assert isinstance(text, str)
    assert isinstance(tokens, list)
    assert isinstance(logprobs, list)

    # Test generate_attn (no entropy/logprob usage)
    text2, seqlist, attns, seqlogprobs, seqentropies = gen.generate_attn("hello", max_length=3, solver="max", use_entropy=False, use_logprob=False)
    assert isinstance(text2, str)
    assert isinstance(seqlist, list)
    assert isinstance(attns, list)
    assert seqlogprobs is None
    assert seqentropies is None
