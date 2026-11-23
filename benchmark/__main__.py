"""Run `python -m benchmark --help` from the repository root."""
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse.csgraph import shortest_path
import torch

from .core import maxsum, score_paths, sta

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def verify_manifest():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    for name, expected in manifest['sha256'].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Integrity mismatch: {name}')
    return len(manifest['sha256'])


def inputs():
    files = sorted((ROOT / 'data/mazes').glob('maze_*.npz'))
    if len(files) != 20:
        raise ValueError('Expected 20 frozen mazes')
    return [(i, load(path)) for i, path in enumerate(files)]


def generate(output):
    """Recreate original mazes and query sampling, without using stored arrays."""
    if output.resolve() in [(ROOT / 'data/mazes').resolve(), (ROOT / 'results/reference').resolve()]:
        raise ValueError('Use a separate output directory; frozen arrays are immutable.')
    spec = importlib.util.spec_from_file_location('upstream_maze', ROOT / 'vendor/pysta/maze_utils.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    output.mkdir(parents=True, exist_ok=True)
    for maze in range(20):
        seed = 4100 + maze
        np.random.seed(seed)
        torch.manual_seed(seed)
        walls = upstream.sample_maze(15)
        adjacency = upstream.compute_adjacency(torch.tensor(walls))[0].numpy()
        distances = shortest_path(adjacency, directed=False, unweighted=True).astype(int)
        rng = np.random.default_rng(seed)
        chosen = []
        for distance in range(1, 31):
            candidates = np.argwhere(distances == distance)
            chosen.extend(candidates[rng.choice(len(candidates), 5, replace=False)])
        starts, goals = np.array(chosen).T
        arrays = dict(walls=walls, adjacency=adjacency, distances=distances, starts=starts, goals=goals)
        reference = load(ROOT / f'data/mazes/maze_{maze:03d}.npz')
        for name, array in arrays.items():
            if not np.array_equal(array, reference[name]):
                raise ValueError(f'Regenerated input mismatch: maze {maze}, {name}')
        np.savez_compressed(output / f'maze_{maze:03d}.npz', **arrays)
    print('Regenerated all 20 mazes and 3,000 pairs: exact array match.')


def run(output, threads):
    if output.resolve() in [(ROOT / 'data/mazes').resolve(), (ROOT / 'results/reference').resolve()]:
        raise ValueError('Use a new output directory; reference outputs are immutable.')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    timings = []
    for maze, x in inputs():
        start = time.perf_counter()
        snapshots = sta(x['adjacency'], x['starts'], x['goals'])
        sta_seconds = time.perf_counter() - start
        start = time.perf_counter()
        paths, scores = maxsum(x['adjacency'], x['starts'], x['goals'])
        maxsum_seconds = time.perf_counter() - start
        arrays = dict(maxsum_paths=paths, maxsum_scores=scores)
        for iteration, (path, peak) in snapshots.items():
            arrays[f'sta_{iteration}_paths'] = path
            arrays[f'sta_{iteration}_peaks'] = peak
        np.savez_compressed(output / f'maze_{maze:03d}.npz', **arrays)
        timings.append(dict(maze=maze, sta_seconds=sta_seconds, maxsum_seconds=maxsum_seconds))
        print(f'Maze {maze + 1}/20: STA {sta_seconds:.2f}s; max-sum {maxsum_seconds:.3f}s', flush=True)
    metadata = dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
                    scipy=scipy.__version__, torch=torch.__version__, torch_threads=threads,
                    per_maze_timings=timings, timing_scope='planner calls only; excludes I/O and evaluation')
    (output / 'run.json').write_text(json.dumps(metadata, indent=2))


def evaluate(results):
    rows = []
    for maze, x in inputs():
        distances = shortest_path(x['adjacency'], directed=False, unweighted=True).astype(int)
        if not np.array_equal(distances, x['distances']):
            raise ValueError(f'Stored distance mismatch: maze {maze}')
        r = load(results / f'maze_{maze:03d}.npz')
        for algorithm, iterations, key in [('STA', 400, 'sta_400'), ('STA', 1500, 'sta_1500'), ('Max-sum belief prop', 30, 'maxsum')]:
            paths = r[f'{key}_paths']
            if paths.shape != (150, 31):
                raise ValueError(f'Bad path shape: maze {maze}, {key}')
            scored = score_paths(paths, x['adjacency'], distances, x['starts'], x['goals'], r.get(f'{key}_peaks'))
            if key == 'maxsum':
                expected = distances[x['starts'], x['goals']]
                if not all(row['shortest'] for row in scored) or not np.array_equal(r['maxsum_scores'], -expected):
                    raise ValueError('Max-sum failed the shortest-path control')
                for p, goal, distance in zip(paths, x['goals'], expected):
                    if not (p[distance:] == goal).all():
                        raise ValueError('Max-sum goal is not absorbing')
            rows.extend(dict(maze=maze, algorithm=algorithm, iterations=iterations, **row) for row in scored)
    return rows


def plot(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = evaluate(results)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'trial_scores.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    bootstrap = np.random.default_rng(42).integers(0, 20, (5000, 20))
    summaries = []
    fig, ax = plt.subplots(figsize=(9, 5.2), layout='constrained')
    ax.spines[['top', 'right']].set_visible(False)
    styles = [('STA', 400, '--', '#237c9a', 'STA: original finite inputs, 400 updates'),
              ('STA', 1500, '-', '#237c9a', 'STA: original finite inputs, 1,500 updates'),
              ('Max-sum belief prop', 30, '-', '#8d55ae', 'Max-sum belief prop')]
    for algorithm, iteration, line, color, label in styles:
        rates = np.zeros((20, 30))
        counts = np.zeros((20, 30))
        for row in rows:
            if row['algorithm'] == algorithm and row['iterations'] == iteration:
                rates[row['maze'], row['distance'] - 1] += row['shortest']
                counts[row['maze'], row['distance'] - 1] += 1
        if not (counts == 5).all():
            raise ValueError('Expected five queries per distance per maze')
        rates /= counts
        mean = rates.mean(0)
        boot = rates[bootstrap].mean(1)
        lower, upper = np.quantile(boot, [.025, .975], axis=0)
        for d in range(30):
            summaries.append(dict(algorithm=algorithm, iterations=iteration, distance=d + 1, trials=100,
                                  success=float(mean[d]), ci_lower=float(lower[d]), ci_upper=float(upper[d])))
        ax.plot(np.arange(1, 31), 100 * mean, line, color=color, lw=2, label=label,
                marker='o' if algorithm != 'STA' else None, markersize=3.5)
        ax.fill_between(np.arange(1, 31), 100 * lower, 100 * upper, color=color, alpha=.12)
    ax.set(xlabel='Shortest-path distance (maze moves)', ylabel='Shortest legal plan success (%)',
           xlim=(1, 30), ylim=(-2, 107), title='Planning on 15x15 mazes')
    ax.grid(alpha=.15)
    ax.legend(loc='center right', fontsize=9)
    fig.supxlabel('20 mazes · 100 pairs per distance · horizon 30\n'
                  'Max-sum uses hard start/goal constraints; STA retains the original benchmark inputs.', fontsize=9)
    fig.savefig(output / 'planning-15x15-mazes.png', dpi=180)
    fig.savefig(output / 'planning-15x15-mazes.pdf')
    plt.close(fig)
    with (output / 'success-vs-distance.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print(f'Figure and tables written to {output}')


def compare(results, output):
    reference = evaluate(ROOT / 'results/reference')
    actual = evaluate(results)
    differences = [dict(reference=a, rerun=b) for a, b in zip(reference, actual) if a != b]
    array_differences = []
    for maze in range(20):
        ref = load(ROOT / f'results/reference/maze_{maze:03d}.npz')
        rerun = load(results / f'maze_{maze:03d}.npz')
        for key in ref:
            if not np.array_equal(ref[key], rerun[key]):
                array_differences.append(dict(maze=maze, field=key, unequal_elements=int(np.sum(ref[key] != rerun[key])),
                                              max_absolute_difference=float(np.max(np.abs(ref[key] - rerun[key])))))
    report = dict(scored_trials=len(actual), score_differences=differences, array_differences=array_differences,
                  scores_match=not differences)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f'Score differences: {len(differences)}; differing arrays: {len(array_differences)}')
    if differences:
        raise SystemExit('Rerun differs from reference; see comparison report.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['run', 'plot', 'generate-inputs', 'compare', 'verify']:
        p = commands.add_parser(name)
        if name in ['plot', 'compare']:
            p.add_argument('--results', type=Path, default=ROOT / ('scratch/rerun' if name == 'compare' else 'results/reference'))
        if name != 'verify':
            defaults = {'run': 'scratch/rerun', 'plot': 'figures', 'generate-inputs': 'scratch/generated-inputs',
                        'compare': 'scratch/comparison.json'}
            p.add_argument('--output', type=Path, default=ROOT / defaults[name])
        if name == 'run':
            p.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    checked = verify_manifest()
    if args.command == 'run': run(args.output, args.threads)
    elif args.command == 'plot': plot(args.results, args.output)
    elif args.command == 'generate-inputs': generate(args.output)
    elif args.command == 'compare': compare(args.results, args.output)
    else:
        rows = evaluate(ROOT / 'results/reference')
        print(f'Integrity verified for {checked} files; rescored {len(rows)} method/query results.')


if __name__ == '__main__':
    main()
