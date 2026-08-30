

from termcolor import colored
import time

def format_time_color(t):
    if t < 1e-6:
        return colored(f"{t*1e9:.2f} ns", "cyan")
    elif t < 1e-3:
        return colored(f"{t*1e6:.2f} µs", "green")
    elif t < 1:
        return colored(f"{t*1e3:.2f} ms", "yellow")
    else:
        return colored(f"{t:.4f} s", "red")


