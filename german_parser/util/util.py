def get_str_after_underscore(text: str):
    return text.split("_")[1]

def get_int_after_underscore(text: str, default: int | None = None) -> int | None:
    res = None
    try:
        res = int(get_str_after_underscore(text))
    except Exception:
        res = default
    return res

def is_pairwise_disjoint(*args: set) -> bool:
    # sets are pairwise disjoint if and only if the length of the union of all sets is equal to the sum of the lengths of all sets
    return len(set.union(*args)) == sum(map(len, args))
