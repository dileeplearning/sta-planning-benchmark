# Experiment specification and limitations

## Inputs and sampling

Twenty independent 15×15 mazes are generated with the upstream `sample_maze(15)`
routine, using NumPy seeds 4100–4119. It builds a depth-first-search maze and
removes 36 additional walls. There are 225 spatial locations, with self-transitions
allowed. For each maze, the original generator computes graph distances and uses
`default_rng(seed)` to choose five distinct ordered start–goal pairs without
replacement at each exact distance from 1 through 30. There are 150 queries per
maze and 3,000 queries total. Both methods use exactly these same queries.

Distances are used for stratified query selection and subsequent evaluation.
Neither planner receives them, an optimal route, or a distance-dependent reward
schedule. The fixed horizon is 30 actions / 31 time modules for every query.

## STA configuration

The planner is our block evaluation of the upstream first-action STA recurrence,
not a direct large-scale instantiation of the upstream class. Its 31 modules each
contain 225 units (6,975 total). Forward and backward weights in each time-neighbor
block equal the maze adjacency. Both activity noise and weight perturbations,
**including the negative weight offset**, are absent. There is no observation
noise, no trained weight and no time-shift component before this first action.

The initial-location module has input 20 at the start and 0 elsewhere. Every
future module has reward +0.6 at the goal and −0.6 elsewhere. All these inputs are
multiplied by β=6 and normalized in log space over locations. The initial start
signal is therefore a **finite preference**, not hard evidence. β=6 is an explicit
benchmark choice; the upstream STA class default is β=9.

Log activities start uniformly at −log(225). At each update, the forward and
backward sums use neighboring full activities (not BP cavity messages), are
floored at exp(−100), and enter the drive through their logarithms. The Euler
update has τ=50. Log activities are normalized over each module and clipped to
[−100,0]. We record 400 updates (8τ) and 1,500 updates (30τ), with float32 PyTorch
arithmetic. No convergence claim follows from these fixed update budgets.

## Max-sum configuration

The exact planner uses a time-unrolled chain with the same 30-action horizon.
The supplied start is hard evidence. Only maze-adjacency transitions are allowed,
the goal is absorbing, and the terminal state is required to be the goal. Each
transition from a non-goal state costs 1; staying at the goal costs 0. Maximizing
trajectory score therefore minimizes first-arrival time. Illegal transitions
have score −∞. A forward max-sum pass records the maximizing predecessor for each
state at each time, and backtracing from the terminal goal returns one coherent
joint route. It uses float64 NumPy scores. No numerical floor softens constraints.

This is max-sum / max-product inference with backtracing, not sum-product followed
by independent marginal argmax. The figure's label is “Max-sum belief prop”.

## Evaluation

STA is decoded by selecting each module's most active location independently.
A shortest-plan success requires a correct start and a legal route reaching the
goal after exactly the graph shortest distance, with distance-to-go decreasing by
one along the prefix. Scoring stops at arrival: activity afterward does not affect
this figure. This convention is retained from the original benchmark. Additional
trial columns record any legal goal-reaching prefix within the horizon and a
STA-only confidence condition requiring peak activity ≥0.3 through arrival.
Confidence is not defined for max-sum and is left empty in that column.

Max-sum paths use the same shortest-prefix criterion. They are additionally
checked for an absorbing goal suffix and a score equal to minus graph distance.
Maze distances are independently recomputed during evaluation using SciPy's graph
shortest-path routine, rather than trusting the stored distance arrays.

Rates pool five queries from each of 20 mazes at every distance. Pointwise 95%
percentile bootstrap intervals resample the **20 mazes**, not individual queries,
using 5,000 resamples and seed 42. For this figure the per-distance rates are all
zero or one, so the empirical bootstrap intervals collapse. This does not prove
population success probabilities are exactly zero or one. Lines connect integer
distance samples; no interpolation experiment is implied.

## What this does and does not establish

The result tests one STA parameter setting, maze distribution, initialization,
horizon and decoding procedure against an exact task-solving reference. It does
not establish that every STA variant fails, that failure persists at arbitrarily
long settling times, or that recurrent circuits generally cannot plan.

The constraint encodings, numerical floors, reward formulations and decoders
**differ between the methods**. A same-factor, same-readout algorithmic comparison
would be a different experiment. The distinct strengths of hard conditioning,
maximization, and joint backtracing cannot be attributed to just one of them by
this figure. Likewise, a near-perfect first action and a coherent complete plan
are different criteria. The figure concerns the latter.

The fixed 30-action horizon bounds the problem size. A distance-30 query has no
slack: every feasible goal-reaching trajectory within the horizon is shortest.
The benchmark does not study generalization, learning a maze, or closed-loop
navigation. Noisy, parallel-timing, sum-product, and interrupted hard-constrained
STA follow-ups from the exploratory workspace are deliberately not included.
