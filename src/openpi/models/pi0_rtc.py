import functools
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0Rtc(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0RtcConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

        self.hist_horizon = config.hist_horizon
        self.max_delay = config.max_delay

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], int, int]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        all_image_tokens = dict()

        for name in obs.images:
            image = obs.images[name]
            bs, T = image.shape[0], image.shape[1]
            image = image.reshape(bs*T, image.shape[2], image.shape[3], image.shape[4])
            image_tokens, _ = self.PaliGemma.img(image, train=False)
            image_tokens = image_tokens.reshape(bs, T, image_tokens.shape[1], image_tokens.shape[2])
    
            all_image_tokens[name] = image_tokens

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            num_text_tokens = tokenized_inputs.shape[1]
        else:
            num_text_tokens = 0

        for t in range(T):
            visual_tokens = list()
            visual_input_mask = list()
            for name in obs.images:
                visual_tokens.append(all_image_tokens[name][:, t])

                visual_input_mask.append(einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=visual_tokens[0].shape[1],
                ))

            visual_tokens = jnp.concatenate(visual_tokens, axis=1)
            visual_input_mask = jnp.concatenate(visual_input_mask, axis=1)

            multimodal_tokens = jnp.concatenate([visual_tokens, tokenized_inputs], axis=1)
            multimodal_input_mask = jnp.concatenate([visual_input_mask, obs.tokenized_prompt_mask], axis=1)

            # image tokens attend to each other
            ar_mask += [True] + [False] * (visual_tokens.shape[1] - 1)

            # full attention between image and language inputs
            ar_mask += [False] * (tokenized_inputs.shape[1])

            tokens.append(multimodal_tokens)
            input_mask.append(multimodal_input_mask)

        num_visual_tokens = tokens[0].shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1).reshape(bs, -1)
    
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, num_visual_tokens, T

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b ah"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b ah emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        # time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = jax.vmap(
            functools.partial(
                posemb_sincos, embedding_dim=self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
            )
        )(timestep)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, mask_rng, delay_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        b, ah, ad = actions.shape
        batch_shape = actions.shape[:-2]

        # Generate random delay for each batch
        delay = jax.random.randint(delay_rng, (b,), 0, self.max_delay)
        # Create mask where positions < delay are True
        prefix_action_mask = jnp.arange(ah)[None, :] < delay[:, None]  # (b, ah)

        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (b, 1)) * 0.999 + 0.001
        time = jnp.broadcast_to(time, (b, ah))

        # Set time to 0 where mask is True (prefix positions get ground truth actions)
        time = jnp.where(prefix_action_mask, 0.0, time)  # (b, ah)

        x_t = time[..., None] * noise + (1 - time[..., None]) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask, num_img_tokens, T = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)

        mask_num = jax.random.randint(mask_rng, (), 0, T)
        mask_tokens = mask_num * num_img_tokens
        seq_len = input_mask.shape[1]
        indices = jnp.arange(seq_len)
        
        to_mask = ~(indices < mask_tokens)
        input_mask = input_mask & to_mask[None, :]

        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # (b, ah)
        postfix_action_mask = jnp.logical_not(prefix_action_mask)  # (b, ah)
        loss = jnp.sum(loss * postfix_action_mask) / (jnp.sum(postfix_action_mask) + 1e-8)

        return loss
    
    @at.typecheck
    def embed_prefix_infer(
        self, obs: _model.Observation, memory_tokens
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], int, int]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        all_image_tokens = dict()

        for name in obs.images:
            image = obs.images[name]
            bs, T = image.shape[0], image.shape[1]
            image = image.reshape(bs*T, image.shape[2], image.shape[3], image.shape[4])
            image_tokens, _ = self.PaliGemma.img(image, train=False)
            image_tokens = image_tokens.reshape(bs, T, image_tokens.shape[1], image_tokens.shape[2])
    
            all_image_tokens[name] = image_tokens

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            num_text_tokens = tokenized_inputs.shape[1]
        else:
            num_text_tokens = 0

        for t in range(T):
            visual_tokens = list()
            visual_input_mask = list()
            for name in obs.images:
                visual_tokens.append(all_image_tokens[name][:, t])

                visual_input_mask.append(einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=visual_tokens[0].shape[1],
                ))

            visual_tokens = jnp.concatenate(visual_tokens, axis=1)
            visual_input_mask = jnp.concatenate(visual_input_mask, axis=1)

            multimodal_tokens = jnp.concatenate([visual_tokens, tokenized_inputs], axis=1)
            multimodal_input_mask = jnp.concatenate([visual_input_mask, obs.tokenized_prompt_mask], axis=1)

            # image tokens attend to each other
            ar_mask += [True] + [False] * (visual_tokens.shape[1] - 1)

            # full attention between image and language inputs
            ar_mask += [False] * (tokenized_inputs.shape[1])

            tokens.append(multimodal_tokens)
            input_mask.append(multimodal_input_mask)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1).reshape(bs, -1)
    
        num_visual_tokens = tokens.shape[1]

        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, num_visual_tokens, num_text_tokens

    def reset_memory(self, memory):
        memory["memory_tokens"] = None
        memory["memory_kv_cache"] = None
        memory["memory_prefix_mask"] = None

    @at.typecheck
    def save_memory(
        self, prefix_tokens: at.Float[at.Array, "b s1 emb"], prefix_mask: at.Bool[at.Array, "b s2"], num_visual_tokens: int, num_text_tokens: int, kv_cache, memory: dict
    ):
        memory["memory_tokens"] = prefix_tokens

        memory_kv_cache = list()
        for cache in kv_cache:
            memory_kv_cache.append(cache)

        memory["memory_kv_cache"] = tuple(memory_kv_cache)
        memory["memory_prefix_mask"] = prefix_mask

    def fetch_memory(self, memory):
        memory_tokens = memory["memory_tokens"]
        memory_kv_cache = memory["memory_kv_cache"]
        memory_prefix_mask = memory["memory_prefix_mask"]
        return None, memory_kv_cache, memory_prefix_mask

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        step: at.Int[at.Array, ""] = None,
        delay: at.Int[at.Array, "b"] | None = None,
        action_prefix: at.Float[at.Array, "b ah ad"] | None = None,
        memory: dict = None
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        memory_tokens, memory_kv_cache, memory_prefix_mask = self.fetch_memory(memory)

        if delay is None:
            delay = jnp.zeros((batch_size,), dtype=jnp.int32)
            action_prefix = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

        # if jax.process_index() == 0:
        #     jax.debug.print("x1, {}, {}", memory_kv_cache is None, memory_prefix_mask is None)

        prefix_tokens, prefix_mask, prefix_ar_mask, num_visual_tokens, num_text_tokens = self.embed_prefix_infer(observation, memory_tokens)

        # prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        if memory_prefix_mask is not None:
            memory_prefix_attn_mask = einops.repeat(memory_prefix_mask, "b p -> b s p", s=prefix_mask.shape[1])
            prefix_attn_mask = jnp.concatenate([memory_prefix_attn_mask, prefix_attn_mask], axis=-1)
            prefix_mask = jnp.concatenate([memory_prefix_mask, prefix_mask], axis=-1)
        # memory_kv_cache = None
            # import pdb;pdb.set_trace()
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            positions = positions[..., memory_prefix_mask.shape[-1]:] 
        else:
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions, kv_cache=memory_kv_cache)
        self.save_memory(prefix_tokens, prefix_mask, num_visual_tokens, num_text_tokens, kv_cache, memory)

        prefix_action_mask = jnp.arange(self.action_horizon)[None, :] < delay[:, None]  # (b, ah)

        def step(carry):
            x_t, time = carry
            x_t = jnp.where(prefix_action_mask[..., None], action_prefix, x_t)
            time_ = jnp.broadcast_to(time, batch_size)  # (b, )
            time_ = jnp.where(prefix_action_mask, 0.0, time_[:, None])  # (b, ah)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            # assert full_attn_mask.shape == (
            #     batch_size,
            #     suffix_tokens.shape[1],
            #     prefix_tokens.shape[1] + suffix_tokens.shape[1],
            # )
            # print(full_attn_mask.shape)
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0, memory