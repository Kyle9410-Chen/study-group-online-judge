import torch
import math
import time
from torch import nn
from transformers import GPT2Tokenizer, AutoModelForCausalLM


class Config:
    def __init__(self):
        self.vocabulary_size = 50257
        self.dimension = 768
        self.n_position = 1024
        self.n_layer = 12
        self.n_head = 12
        self.layer_norm_eps = 1e-5


class Conv1D(nn.Module):
    """Linear layer storing weight as (in, out) and using addmm, same as HF GPT-2,
    so fp16 rounding matches the reference bit for bit."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor):
        size_out = x.size()[:-1] + (self.out_features,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


class SelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention = Conv1D(config.dimension, 3 * config.dimension)
        self.projection = Conv1D(config.dimension, config.dimension)
        self.n_head = config.n_head
        self.dimension = config.dimension

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        batch_size, sequence_length, dimension = x.size()

        qkv: torch.Tensor = self.attention(x)
        query, key, value = qkv.split(self.dimension, dim=2)

        query = query.view(
            batch_size, sequence_length, self.n_head, dimension // self.n_head
        ).transpose(1, 2)
        key = key.view(
            batch_size, sequence_length, self.n_head, dimension // self.n_head
        ).transpose(1, 2)
        value = value.view(
            batch_size, sequence_length, self.n_head, dimension // self.n_head
        ).transpose(1, 2)

        # Boolean mask (True = attend): causal AND not padding, the same mask HF passes to SDPA
        mask = torch.tril(
            torch.ones(
                (sequence_length, sequence_length), dtype=torch.bool, device=x.device
            )
        ).view(1, 1, sequence_length, sequence_length)
        if attention_mask is not None:
            mask = mask & attention_mask.bool().view(batch_size, 1, 1, sequence_length)

        y = nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, dimension)

        return self.projection(y)


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.full_connect = Conv1D(config.dimension, 4 * config.dimension)
        self.full_connect_projection = Conv1D(4 * config.dimension, config.dimension)

    def activation(self, x: torch.Tensor):
        # HF "gelu_new" written out; nn.GELU(approximate="tanh") rounds differently in fp16
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )

    def forward(self, x: torch.Tensor):
        return self.full_connect_projection(self.activation(self.full_connect(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.dimension, eps=config.layer_norm_eps)
        self.attention = SelfAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.dimension, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask: torch.Tensor = None):
        x = x + self.attention(self.layer_norm1(x), attention_mask=attention_mask)
        return x + self.mlp(self.layer_norm2(x))


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.dimension)
        self.position_embedding = nn.Embedding(config.n_position, config.dimension)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.layer_norm = nn.LayerNorm(config.dimension, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.dimension, config.vocabulary_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)

        for block in self.layers:
            x = block(x, attention_mask=attention_mask)

        x = self.layer_norm(x)
        return self.lm_head(x)


def gpt2_complete(
    input: list[str],
    max_seq_length: int = 1024,
) -> tuple[list[str], torch.Tensor]:
    """Generate greedy completions with a from-scratch GPT-2 Small implementation.

    Load pretrained GPT-2 Small weights into manually implemented transformer
    blocks. Generate for the entire batch at once, choosing the highest-logit
    token for every unfinished sequence at each step. Stop each sequence at EOS
    or max_seq_length total tokens, including the prompt.

    Return newly generated text for each prompt and a tensor of pre-selection
    logits shaped (batch_size, decoding_steps, 50257). Fill logits with zero
    after a row has finished while other rows continue.
    """

    device = torch.device("cpu")
    config = Config()
    # CI verifies against the HF reference in fp16, so run in fp16 to match its numerics
    dtype = torch.float16
    model = Model(config).to(device=device, dtype=dtype)

    # Load Hugging Face
    hugging_face_model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2", dtype=dtype
    ).to(device)
    hugging_face_state_dict = hugging_face_model.state_dict()
    custom_state_dict = model.state_dict()

    custom_state_dict["lm_head.weight"].copy_(
        hugging_face_state_dict["transformer.wte.weight"]
    )
    for name, param in hugging_face_state_dict.items():
        if name.endswith(".attn.masked_bias") or name.endswith(".attn.bias"):
            continue

        custom_name = name.replace("transformer.", "")
        custom_name = custom_name.replace("wte", "token_embedding")
        custom_name = custom_name.replace("wpe", "position_embedding")
        custom_name = custom_name.replace("h.", "layers.")
        custom_name = custom_name.replace("ln_1", "layer_norm1")
        custom_name = custom_name.replace("ln_2", "layer_norm2")
        custom_name = custom_name.replace("ln_f", "layer_norm")
        custom_name = custom_name.replace("attn.c_attn", "attention.attention")
        custom_name = custom_name.replace("attn.c_proj", "attention.projection")
        custom_name = custom_name.replace("mlp.c_fc", "mlp.full_connect")
        custom_name = custom_name.replace("mlp.c_proj", "mlp.full_connect_projection")

        if custom_name in custom_state_dict:
            with torch.no_grad():
                custom_state_dict[custom_name].copy_(param)

    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(
        "openai-community/gpt2", padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    tokens: torch.Tensor = tokenizer(input, return_tensors="pt", padding=True).to(
        device
    )
    input_ids: torch.Tensor = tokens["input_ids"]
    attention_mask: torch.Tensor = tokens["attention_mask"]

    batch_size = input_ids.size(0)
    lengths = attention_mask.sum(dim=1)
    unfinished = lengths < max_seq_length
    all_logits = []
    generated_tokens = [[] for _ in range(batch_size)]

    while unfinished.any():
        print(f"\rGenerating step {len(all_logits) + 1}...", end="", flush=True)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad():
            logits: torch.Tensor = model(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )

        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits.masked_fill(
            ~unfinished.unsqueeze(-1), 0.0
        )
        all_logits.append(next_token_logits)

        next_tokens: torch.Tensor = torch.argmax(next_token_logits, dim=1)
        next_tokens = next_tokens.masked_fill(~unfinished, tokenizer.pad_token_id)
        lengths = lengths + unfinished.long()
        just_finished = unfinished & (
            (next_tokens == tokenizer.eos_token_id) | (lengths >= max_seq_length)
        )

        for i in range(batch_size):
            if unfinished[i]:
                generated_tokens[i].append(next_tokens[i].item())

        unfinished = unfinished & ~just_finished

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)

        unfinished_old = unfinished | just_finished
        attention_mask = torch.cat(
            [attention_mask, unfinished_old.long().unsqueeze(1)], dim=1
        )

    if all_logits:
        logits_tensor = torch.stack(all_logits, dim=1)
    else:
        logits_tensor = torch.empty(
            (batch_size, 0, config.vocabulary_size), device=device
        )

    completions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return completions, logits_tensor


if __name__ == "__main__":
    completions, logits_tensor = gpt2_complete(["Test", "Hello World"])
    print(completions)
