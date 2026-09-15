import torch

from day5.layers.kv_cache import store_kv_cache


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    block_size = 4
    key = torch.arange(4 * 2 * 3, device=device, dtype=torch.float32).reshape(4, 2, 3)
    value = key + 100
    k_cache = torch.zeros(3, block_size, 2, 3, device=device)
    v_cache = torch.zeros_like(k_cache)
    slots = torch.tensor([9, 1, 6, -1], device=device, dtype=torch.long)

    store_kv_cache(key, value, k_cache, v_cache, slots)
    print(f"device          = {device}")
    print(f"cache shape     = {tuple(k_cache.shape)}")
    print(f"slot mapping    = {slots.tolist()}")
    for token, slot in enumerate(slots.tolist()):
        if slot >= 0:
            block_id, offset = divmod(slot, block_size)
            ok = torch.equal(k_cache[block_id, offset], key[token])
            print(f"token {token} -> block {block_id}, offset {offset}, correct={ok}")
    print(f"written slots   = {(k_cache.abs().sum(dim=(-1, -2)).flatten() != 0).nonzero().flatten().tolist()}")


if __name__ == "__main__":
    main()
