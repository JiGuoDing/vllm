# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.platforms import current_platform

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops  # type: ignore[no-redef]


class PagedAttention:
    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将紧凑存储的 KV 缓存拆分成独立的 key, value 视图，便于后续按头访问或写入
        """

        # 根据元素字节数计算一个分块因子 (例如 FP16 则为 8)，用于对 key 进行子块重排
        x = 16 // kv_cache.element_size()
        # 读取缓存中的分块数量
        num_blocks = kv_cache.shape[1]

        # 把 key 重排为 [块数，头数，head_size // x，序列块，x] 形式，适配底层 paged 缓存的对齐与向量化访问
        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x, -1, x)

        # 把 value 重排为 [块数，头数，head_size，序列块] 形式
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)

        # 返回两个张量视图，不复制数据，仅通过 view 方式提供结构化访问
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
