"""Independent checks of numerical equivalence, path optimization and scoring."""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from benchmark.core import maxsum, score_paths, sta_dynamics

ROOT = Path(__file__).resolve().parents[1]


class PlannerChecks(unittest.TestCase):
    def test_upstream_sta_step(self):
        """Execute the unchanged upstream step body without importing its plotting stack.

        The harness supplies prepared unary inputs and dense weights. This checks
        recurrence equivalence, not upstream environment preprocessing or policy code.
        """
        tree = ast.parse((ROOT / 'vendor/pysta/agents.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpaceTimeAttractor')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
        namespace = {'np': np, 'torch': torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     'vendored-upstream-step', 'exec'), namespace)
        torch.set_num_threads(1)
        torch.manual_seed(7)
        batch, modules, locations = 3, 5, 9
        adjacency = torch.eye(locations)
        for i in range(locations - 1):
            adjacency[i, i + 1] = adjacency[i + 1, i] = 1
        forward = adjacency.expand(modules - 1, -1, -1).clone()
        backward = forward.clone()
        inputs = torch.randn(modules, locations, batch) * 2
        inputs -= inputs.logsumexp(1, keepdim=True)
        total = modules * locations
        dense_f = torch.zeros(batch, total, total)
        dense_b = dense_f.clone()
        for t in range(modules - 1):
            dense_f[:, (t + 1) * locations:(t + 2) * locations, t * locations:(t + 1) * locations] = forward[t]
            dense_b[:, t * locations:(t + 1) * locations, (t + 1) * locations:(t + 2) * locations] = backward[t]
        z = torch.full((batch, modules, locations), -np.log(locations))
        obj = SimpleNamespace(batch=batch, num_locs=locations, iters_per_action=100, tau=50,
                              shift_time=2., env=SimpleNamespace(step_num=0), rec_noise=0.,
                              Win=torch.eye(total), Wrec_fwd=dense_f, Wrec_bwd=dense_b,
                              brec=torch.zeros(total, 1), z=z, z0=z.clone(), r=z.exp(),
                              phi=torch.exp, store_all_activity=False,
                              obs_to_rew_func=lambda observation, flat: observation,
                              calc_policy=lambda: None, sample_action=lambda: None)
        observation = inputs.permute(2, 0, 1).reshape(batch, total, 1)
        namespace['step'](obj, observation)
        actual = sta_dynamics(inputs, forward, backward, snapshots=(100,))[100]
        error = float(np.max(np.abs(actual - obj.r.numpy())))
        self.assertLess(error, 2e-5)
        print(f'Upstream recurrence maximum activity discrepancy: {error:.3g}')

    def test_maxsum_exhaustive(self):
        adjacency = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
        starts = np.array([0, 1, 2])
        goals = np.array([2, 1, 0])
        paths, scores = maxsum(adjacency, starts, goals, horizon=4)
        for b, (start, goal) in enumerate(zip(starts, goals)):
            best = -np.inf
            for tail in itertools.product(range(3), repeat=4):
                path = (start,) + tail
                if path[-1] != goal:
                    continue
                if not all(adjacency[a, c] and (a != goal or c == goal) for a, c in zip(path, path[1:])):
                    continue
                best = max(best, -sum(a != goal for a in path[:-1]))
            self.assertEqual(scores[b], best)
            self.assertEqual(paths[b, 0], start)
            self.assertEqual(paths[b, -1], goal)
            self.assertTrue(all(adjacency[a, c] and (a != goal or c == goal)
                                for a, c in zip(paths[b], paths[b, 1:])))
        paths, scores = maxsum(np.eye(2), np.array([0]), np.array([1]), horizon=3)
        self.assertTrue(np.isneginf(scores[0]))
        self.assertTrue((paths == -1).all())

    def test_score_rejects_teleport_and_accepts_goal_prefix(self):
        adjacency = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
        distances = np.abs(np.arange(3)[:, None] - np.arange(3)[None])
        paths = np.array([[0, 2, 2, 2], [0, 0, 1, 2], [0, 1, 2, 0]])
        rows = score_paths(paths, adjacency, distances, np.array([0, 0, 0]), np.array([2, 2, 2]))
        self.assertEqual([r['shortest'] for r in rows], [False, False, True])
        self.assertEqual([r['valid_reach'] for r in rows], [False, True, True])


if __name__ == '__main__':
    unittest.main()
