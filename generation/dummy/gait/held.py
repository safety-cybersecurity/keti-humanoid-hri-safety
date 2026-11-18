"""Which environments the gait still drives: a released dummy keeps the pose and phase it was let go at."""
import numpy as np


def held_update(old, new, held):
    held = np.asarray(held, bool)
    old, new = np.asarray(old), np.asarray(new)
    mask = held.reshape(held.shape + (1,) * (new.ndim - held.ndim))
    return np.where(mask, new, old)
