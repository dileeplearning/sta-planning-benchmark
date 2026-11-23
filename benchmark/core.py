"""Planner implementations; no precomputed distance is passed to either planner."""
import numpy as np
import torch


def sta(adjacency, starts, goals, horizon=30, beta=6., tau=50., snapshots=(400, 1500)):
    """Block evaluation of upstream STA's first-action, noiseless recurrence.

    Both activity noise and weight perturbations (including their negative bias)
    are absent. Finite start input and the upstream numerical floor are retained.
    """
    batch, locations = len(starts), len(adjacency)
    inputs = torch.full((horizon + 1, locations, batch), -.6)
    inputs[:, goals, torch.arange(batch)] = .6
    inputs[0] = 0.
    inputs[0, starts, torch.arange(batch)] = 20.
    inputs *= beta
    inputs -= inputs.logsumexp(1, keepdim=True)
    weights = torch.tensor(adjacency, dtype=torch.float32).expand(horizon, -1, -1).clone()
    rates = sta_dynamics(inputs, weights, weights, tau, snapshots)
    return {k: (r.argmax(-1), r.max(-1)) for k, r in rates.items()}


def sta_dynamics(inputs, forward, backward, tau=50., snapshots=(400, 1500)):
    z = torch.full_like(inputs, -np.log(inputs.shape[1]))
    floor = torch.tensor(-100.).exp()
    output = {}
    for iteration in range(1, max(snapshots) + 1):
        rates = z.exp()
        fm = torch.full_like(z, -100.)
        bm = torch.full_like(z, -100.)
        fm[1:] = torch.bmm(forward, rates[:-1]).clamp_min(floor).log()
        bm[:-1] = torch.bmm(backward, rates[1:]).clamp_min(floor).log()
        z = (1 - 1 / tau) * z + (1 / tau) * (inputs + fm + bm)
        z = (z - z.logsumexp(1, keepdim=True)).clamp(-100., 0.)
        if iteration in snapshots:
            output[iteration] = z.exp().permute(2, 0, 1).numpy().copy()
    return output


def maxsum(adjacency, starts, goals, horizon=30):
    """Exact max-sum on a time chain, with stored predecessors and backtracing.

    Hard start, absorbing goal, legal edges only, terminal goal required.
    Each transition from a non-goal state costs one unit. Staying at goal is free.
    """
    batch, locations = len(starts), len(adjacency)
    incoming = [np.flatnonzero(adjacency[:, j]) for j in range(locations)]
    width = max(map(len, incoming))
    source = np.zeros((locations, width), dtype=int)
    valid = np.zeros_like(source, dtype=bool)
    for j, neighbors in enumerate(incoming):
        source[j, :len(neighbors)] = neighbors
        valid[j, :len(neighbors)] = True
    allowed = valid[None] & ((source[None] != goals[:, None, None]) |
                            (np.arange(locations)[None, :, None] == goals[:, None, None]))
    cost = (source[None] != goals[:, None, None]).astype(float)
    score = np.full((batch, locations), -np.inf)
    score[np.arange(batch), starts] = 0.
    pointers = np.empty((horizon, batch, locations), dtype=np.int32)
    for t in range(horizon):
        candidates = np.where(allowed, score[:, source] - cost, -np.inf)
        choices = candidates.argmax(axis=2)
        score = candidates.max(axis=2)
        pointers[t] = source[np.arange(locations)[None], choices]
    terminal = score[np.arange(batch), goals]
    paths = np.full((batch, horizon + 1), -1, dtype=np.int32)
    feasible = np.flatnonzero(np.isfinite(terminal))
    paths[feasible, -1] = goals[feasible]
    for t in range(horizon - 1, -1, -1):
        paths[feasible, t] = pointers[t, feasible, paths[feasible, t + 1]]
    return paths, terminal


def score_paths(paths, adjacency, distances, starts, goals, peaks=None):
    """Original benchmark score: truncate a successful plan at its first goal.

    Activity after arrival does not affect the main STA score. A shortest legal
    prefix must have length equal to the independently computed maze distance.
    """
    output = []
    for i, (start, goal) in enumerate(zip(starts, goals)):
        path = paths[i]
        distance = int(distances[start, goal])
        if (path < 0).any() or (path >= len(adjacency)).any():
            shortest = reached = False
        else:
            legal = adjacency[path[:-1], path[1:]] > 0
            shortest = bool(distance < len(path) and path[0] == start and path[distance] == goal
                            and legal[:distance].all()
                            and np.all(distances[path[:distance + 1], goal] == np.arange(distance, -1, -1)))
            reached = bool(path[0] == start and any(legal[:h].all() for h in np.flatnonzero(path == goal)))
        output.append(dict(start=int(start), goal=int(goal), distance=distance,
                           shortest=shortest, valid_reach=reached,
                           confident_shortest=None if peaks is None else bool(shortest and (peaks[i, :distance + 1] >= .3).all())))
    return output
