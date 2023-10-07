import re

from math import floor, ceil
from time import strftime
from torch.nn import Module
from prettytable import PrettyTable

def str_to_newick_str(text: str) -> str:
    quote = "'"
    # return f"'{text.replace(quote, quote * 2)}'"
    # return re.sub(r"([',;:()\[\]\s])", r"_", text)
    return text.replace("'", "QUOTE").replace(",", "COMMA").replace(";", "SEMI").replace(":", "COLON").replace("(", "LBR").replace(")", "RBR").replace("[", "LSQ").replace("]", "RSQ").replace(" ", "SPACE")

def get_str_after_underscore(text: str):
    return text.split("_")[1]

def get_str_before_underscore(text: str):
    return text.split("_")[0]

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

def get_progress_bar(progress: float, num_bars: int = 10):
    left_bars = "█" * max(0, ceil(progress * num_bars) - 1)
    right_bars = "░" * floor(num_bars - progress * num_bars)
    progress_frac_part = progress * num_bars - floor(progress * num_bars)
    middle_bar = "█ ▏▎▍▌▋▊▉"[ceil(progress_frac_part * 8)] if progress != 0 else ""
    return left_bars + middle_bar + right_bars

filename_prefix = f"tiger_model_{strftime('%Y_%m_%d-%I_%M_%S_%p')}"

def get_filename(epoch: int):
    """get filename of model (without extension)

    Args:
        epoch (int): 0-indexed epoch number
    """

    return f"{filename_prefix}_epoch_{epoch + 1}"

def print_module_parameters(module: Module):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue

        params = parameter.numel()
        table.add_row([name, f"{params:_}".replace("_", " ")])

        total_params += params

    print(table)
    print(f"Total trainable params: {f'{total_params:_}'.replace('_', ' ')}")

    return total_params