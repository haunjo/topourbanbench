"""Generate one scene from the Town10 demo map and show what goes in and what comes out.

    python demo.py                    # demo map carla_town10_049
    python demo.py --example          # no generation, draw the example output in examples/topo_ft/
    python demo.py --scene <scene>    # a map you extracted from CARLA into datasets/Synthetic/

Generation is the same as `./scripts/generate_crowdes_trajectories.sh --scene <scene> --randomize`. The demo then
writes three pictures to output/demo/<scene>/:

    input.png    what the generator reads: map image, walkable area, and semantic anchors
    output.png   what it writes: trajectories, group graphs, and expert paths
    replay.gif   the generated scene at 8x speed
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
CROWDES = os.path.join(ROOT, 'topourbanbench', 'generation', 'crowdes')
MAPS = os.path.join(CROWDES, 'datasets', 'Synthetic')
GENERATED = os.path.join(ROOT, 'output', 'generated', 'synthetic', 'topo_ft')
EXAMPLES = os.path.join(ROOT, 'examples', 'topo_ft')

FPS = 20          # frame rate of the generated trajectories
GIF_SPEED = 8     # replay.gif plays the scene this many times faster than real time
GIF_FPS = 10
GIF_SIZE = 640    # replay.gif width in pixels

COLORS = {        # the colors of the README figures
    'queue': '#1565C0',
    'gathering': '#E65100',
    'cohesive': '#2E7D32',
    'none': '#9C27B0',
    'expert': '#D50000',
}
ENTRY_COLOR = '#00E5FF'
LEGEND = [
    ('queue', 'queue (chain $P_n$)'),
    ('gathering', 'gathering (ring $C_n$)'),
    ('cohesive', 'cohesive group (clique $K_n$)'),
    ('none', 'unaffiliated pedestrian'),
    ('expert', 'expert path'),
]
ANCHOR_NAMES = {'bus': 'bus stop', 'bus_stop': 'bus stop', 'food_cart': 'food cart', 'food_cart_dynamic': 'food cart'}
LABEL_STYLE = dict(fontsize=9, color='white', fontweight='bold',
                   path_effects=[patheffects.withStroke(linewidth=2.5, foreground='black')])


def rel(path):
    return os.path.relpath(path, ROOT)


def rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def new_figure(titles, captions):
    fig = Figure(figsize=(12, 5.4), dpi=150)
    axes = fig.subplots(1, 3)
    for ax, title, caption in zip(axes, titles, captions):
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel(caption, fontsize=9, family='monospace', color='#333333')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.19, wspace=0.05)
    return fig, axes


def show_map(ax, image, brightness=1.0):
    """Draw a map image so that plots use the pixel coordinates of the full-size map."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (1024, 1024 * h // w), interpolation=cv2.INTER_AREA)
    ax.imshow((small * brightness).astype(np.uint8), extent=(0, w, h, 0))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)


def graph_edges(rows, groups):
    """Edges of the group graphs between the members present in `rows` (one frame)."""
    pos = {a: (x, y) for a, x, y in zip(rows['agent_id'], rows['x'], rows['y'])}
    for group in groups:
        members = group['member_ids']
        for i, j in group['edges']:
            a, b = members[i], members[j]
            if a in pos and b in pos:
                yield group['group_type'], pos[a], pos[b]


def plot_frame(ax, rows, groups, kinds=('queue', 'gathering', 'cohesive', 'none')):
    for kind, a, b in graph_edges(rows, groups):
        if kind in kinds:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=COLORS[kind], lw=2.2, zorder=2)
    for kind in kinds:
        sub = rows[rows['group_type'] == kind]
        ax.scatter(sub['x'], sub['y'], s=30, color=COLORS[kind], edgecolors='white', linewidths=0.8, zorder=3)


def plot_experts(ax, df):
    halo = [patheffects.Stroke(linewidth=5, foreground='white'), patheffects.Normal()]
    for _, rows in df[df['group_type'] == 'expert'].groupby('agent_id'):
        xy = rows.sort_values('frame')[['x', 'y']].to_numpy()
        ax.plot(xy[:, 0], xy[:, 1], color=COLORS['expert'], lw=2.6, zorder=4, path_effects=halo)
        ax.plot(*xy[0], 'o', ms=9, mfc='white', mec=COLORS['expert'], mew=2.5, zorder=5)
        ax.plot(*xy[-1], 'o', ms=9, mfc=COLORS['expert'], mec='white', mew=1.5, zorder=5)


# ---------------------------------------------------------------------------------------------------------------------
# 1. Input
# ---------------------------------------------------------------------------------------------------------------------

def show_input(scene, out_dir):
    files = {
        'map': f'image_terrain/{scene}_bg.png',
        'walkable': f'gt/{scene}_walkable.png',
        'entry': f'gt/{scene}_appearance_density.png',
        'anchors': f'metadata/{scene}_meta.json',
    }
    bg = cv2.cvtColor(cv2.imread(os.path.join(MAPS, files['map'])), cv2.COLOR_BGR2RGB)
    walkable = cv2.imread(os.path.join(MAPS, files['walkable']), cv2.IMREAD_GRAYSCALE) > 127
    entry = cv2.imread(os.path.join(MAPS, files['entry']), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    with open(os.path.join(MAPS, files['anchors'])) as f:
        meta = json.load(f)
    h, w = bg.shape[:2]
    m_per_px = meta['ortho_size'] / w
    queues = meta['scenarios'].get('queue', [])
    gatherings = meta['scenarios'].get('gathering', [])
    sources = Counter(ANCHOR_NAMES.get(q.get('source'), 'other') for q in queues)
    source_text = ', '.join(f'{n} {name}' + ('s' if n > 1 else '') for name, n in sources.items())

    print(f'\n[1/3] Input  ({rel(MAPS)}/)')
    for key, what in [
        ('map', f'map image, {w} x {h} px = {w * m_per_px:.0f} m x {h * m_per_px:.0f} m'),
        ('walkable', 'where pedestrians can walk'),
        ('entry', 'where pedestrians enter the map'),
        ('anchors', f'semantic anchors: {len(queues)} queue lines ({source_text}), {len(gatherings)} gathering spots'),
    ]:
        print(f'  {files[key]:<44} {what}')
    print('  The generator also reads configs/scenarios.yaml (archetype parameters) and the CrowdES ZARA2 model.')

    fig, axes = new_figure(
        ['Map image', 'Walkable area', 'Semantic anchors'],
        [files['map'], f"{files['walkable']}\n{files['entry']}",
         f"{files['anchors']}\n(cohesive groups need no anchor)"])
    show_map(axes[0], bg)

    walk = bg.astype(np.float32)
    walk[~walkable] *= 0.3
    alpha = entry[..., None]
    walk = walk * (1 - alpha) + np.array(rgb(ENTRY_COLOR), np.float32) * alpha
    show_map(axes[1], walk)

    show_map(axes[2], bg, brightness=0.6)
    for q in queues:
        head = np.array(q['goal_point'], float)
        tail = head + np.array(q['direction'], float) * q.get('walkable_length_px', q.get('length_px', 0))
        axes[2].plot([head[0], tail[0]], [head[1], tail[1]], color=COLORS['queue'], lw=6, alpha=0.9,
                     solid_capstyle='round')
        axes[2].plot(*head, 'o', ms=10, color=COLORS['queue'], mec='white', mew=1.5)
        right_edge = head[0] > 0.7 * w
        axes[2].annotate(ANCHOR_NAMES.get(q.get('source'), 'queue'), head, xytext=(-8 if right_edge else 8, 6),
                         textcoords='offset points', ha='right' if right_edge else 'left', **LABEL_STYLE)
    for g in gatherings:
        axes[2].add_patch(Circle(g['center'], g.get('spawn_radius', 80), fc=COLORS['gathering'], alpha=0.55))
        axes[2].add_patch(Circle(g['center'], g.get('spawn_radius', 80), fill=False, ec='white', lw=1.5))

    fig.legend(handles=[
        Patch(fc='white', ec='#888888', label='walkable area'),
        Patch(fc=ENTRY_COLOR, label='entry area'),
        Line2D([], [], color=COLORS['queue'], lw=5, marker='o', ms=8, mec='white', label='queue line (dot: head)'),
        Patch(fc=COLORS['gathering'], label='gathering spot'),
    ], loc='lower center', ncol=4, frameon=False, fontsize=10.5)
    fig.suptitle(f'Input: {scene}', fontsize=14)
    path = os.path.join(out_dir, 'input.png')
    fig.savefig(path)
    print(f'  -> {rel(path)}')
    return bg, m_per_px


# ---------------------------------------------------------------------------------------------------------------------
# 2. Generate
# ---------------------------------------------------------------------------------------------------------------------

def generate(scene, force, out_dir):
    stem = os.path.join(GENERATED, f'{scene}-crowdes-synthetic-0')
    print('\n[2/3] Generate  (CrowdES crowd, group graphs, and expert paths)')
    if os.path.exists(stem + '.csv') and not force:
        print(f'  Found {rel(stem)}.csv, so generation is skipped. Pass --force to generate it again.')
        return stem
    if not os.path.isdir(os.path.join(CROWDES, 'checkpoints', 'zara2')):
        sys.exit(f'  The CrowdES weights are missing ({rel(CROWDES)}/checkpoints/zara2/).\n'
                 '  Download them as described under "Pretrained Weights" in README.md, or run\n'
                 '  python demo.py --example to draw the example output without generating.')

    cmd = [sys.executable, 'trainval.py', '--synthetic', '--scene', scene, '--randomize']
    if force:
        cmd.append('--force')
    log_path = os.path.join(out_dir, 'generation.log')
    print(f'  Same as ./scripts/generate_crowdes_trajectories.sh --scene {scene} --randomize')
    print(f'  Log: {rel(log_path)}')
    start = time.time()
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(cmd, cwd=CROWDES, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            print(f'\r  Generating ... {time.time() - start:.0f} s', end='', flush=True)
            time.sleep(1)
    print(f'\r  Generated in {time.time() - start:.0f} s.     ')
    if proc.returncode != 0 or not os.path.exists(stem + '.csv'):
        with open(log_path) as f:
            tail = f.readlines()[-20:]
        sys.exit('  Generation failed. The last lines of the log:\n' + ''.join('    ' + line for line in tail))
    return stem


# ---------------------------------------------------------------------------------------------------------------------
# 3. Output
# ---------------------------------------------------------------------------------------------------------------------

def pick_snapshot(df, groups):
    """The time (in whole seconds) at which the most group graphs and expert paths are in the scene."""
    best, best_score = 0, -1
    for frame, rows in df[df['frame'] % FPS == 0].groupby('frame'):
        present = set(rows['agent_id'])
        score = sum(len(present.intersection(g['member_ids'])) >= 2 for g in groups)
        score += (rows['group_type'] == 'expert').sum()
        if score > best_score:
            best, best_score = frame, score
    return best


def show_output(scene, stem, bg, m_per_px, out_dir):
    csv_path = stem + '.csv'
    json_path = stem + '_group_graph.json'
    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        groups = json.load(f)['groups']
    agents = df.groupby('agent_id')['group_type'].first()
    n_agents = agents.value_counts()
    n_frames = int(df['frame'].max()) + 1

    print(f'\n[3/3] Output  ({rel(os.path.dirname(csv_path))}/)')
    print(f'  {os.path.basename(csv_path)}')
    print(f'    one row per agent and frame: {n_frames} frames ({n_frames / FPS:.0f} s at {FPS} Hz), '
          f'{len(agents)} agents')
    for line in df.head(3).to_string(index=False).splitlines():
        print('    ' + line)
    print(f'    x and y are pixels of the map image ({m_per_px:.4f} m per pixel).')
    print('    agents per group_type: ' + ', '.join(f'{k} {n_agents.get(k, 0)}' for k, _ in LEGEND))
    print(f'  {os.path.basename(json_path)}')
    print(f'    {len(groups)} groups. The largest group of each type (edges are index pairs into member_ids):')
    for kind in ['queue', 'gathering', 'cohesive']:
        g = max((g for g in groups if g['group_type'] == kind), key=lambda g: len(g['member_ids']), default=None)
        if g is not None:
            print(f"    group {g['group_id']:<3} {kind:<10} {g['topology']:<15} members {g['member_ids']}  "
                  f"edges {g['edges']}")
    experts = []
    for agent_id, rows in df[df['group_type'] == 'expert'].groupby('agent_id'):
        xy = rows.sort_values('frame')[['x', 'y']].to_numpy()
        length = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() * m_per_px
        experts.append(f'agent {agent_id} ({length:.1f} m, {len(rows) / FPS:.0f} s)')
    print('  Expert paths (rows with group_type "expert"): ' + (', '.join(experts) or 'none'))

    frame = pick_snapshot(df, groups)
    fig, axes = new_figure(
        ['Trajectories', f'Group graphs at t = {frame / FPS:.0f} s', 'Expert paths'],
        [f'{os.path.basename(csv_path)}\nx, y of every agent, colored by group_type',
         f'{os.path.basename(json_path)}\nmember_ids and edges of each group',
         'rows of the CSV with group_type "expert"\n(hollow dot: start, filled dot: goal)'])
    show_map(axes[0], bg, brightness=0.75)
    for agent_id, rows in df[df['group_type'] != 'expert'].groupby('agent_id'):
        xy = rows.sort_values('frame')[['x', 'y']].to_numpy()
        color = COLORS.get(rows['group_type'].iloc[0], COLORS['none'])
        axes[0].plot(xy[:, 0], xy[:, 1], color=color, lw=1.2, alpha=0.85)
        axes[0].plot(*xy[-1], 'o', ms=3, color=color)
    rows = df[df['frame'] == frame]
    show_map(axes[1], bg, brightness=0.75)
    plot_frame(axes[1], rows, groups)
    ex = rows[rows['group_type'] == 'expert']
    axes[1].scatter(ex['x'], ex['y'], s=60, color=COLORS['expert'], edgecolors='white', linewidths=1.2, zorder=4)
    show_map(axes[2], bg, brightness=0.75)
    plot_frame(axes[2], rows, groups, kinds=('queue', 'gathering'))
    plot_experts(axes[2], df)
    fig.legend(handles=[Line2D([], [], color=COLORS[k], lw=2.2, marker='o', ms=7, mec='white', label=label)
                        for k, label in LEGEND], loc='lower center', ncol=5, frameon=False, fontsize=10)
    fig.suptitle(f'Output: {scene}', fontsize=14)
    path = os.path.join(out_dir, 'output.png')
    fig.savefig(path)
    print(f'  -> {rel(path)}')

    path = os.path.join(out_dir, 'replay.gif')
    write_replay(df, groups, bg, path)
    print(f'  -> {rel(path)}')


def legend_strip(width):
    strip = np.full((34, width, 3), 255, np.uint8)
    x = 12
    for kind, label in [('queue', 'queue'), ('gathering', 'gathering'), ('cohesive', 'cohesive'),
                        ('none', 'unaffiliated'), ('expert', 'expert')]:
        cv2.circle(strip, (x, 17), 6, rgb(COLORS[kind]), -1, cv2.LINE_AA)
        cv2.putText(strip, label, (x + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        x += 24 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 18
    return strip


def write_replay(df, groups, bg, path):
    scale = GIF_SIZE / bg.shape[1]
    base = cv2.resize(bg, (GIF_SIZE, round(bg.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    strip = legend_strip(GIF_SIZE)
    by_frame = {frame: rows for frame, rows in df.groupby('frame')}
    experts = [rows.sort_values('frame') for _, rows in df[df['group_type'] == 'expert'].groupby('agent_id')]

    def px(point):
        return int(round(point[0] * scale)), int(round(point[1] * scale))

    images = []
    for frame in range(0, int(df['frame'].max()) + 1, FPS * GIF_SPEED // GIF_FPS):
        img = base.copy()
        rows = by_frame.get(frame, df.iloc[:0])
        for kind, a, b in graph_edges(rows, groups):
            cv2.line(img, px(a), px(b), rgb(COLORS[kind]), 2, cv2.LINE_AA)
        for kind in ['none', 'cohesive', 'gathering', 'queue']:
            for point in rows.loc[rows['group_type'] == kind, ['x', 'y']].to_numpy():
                cv2.circle(img, px(point), 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, px(point), 5, rgb(COLORS[kind]), -1, cv2.LINE_AA)
        for expert in experts:
            past = expert[expert['frame'] <= frame][['x', 'y']].to_numpy()
            if len(past) == 0:
                continue
            trail = np.array([px(p) for p in past], np.int32)
            cv2.polylines(img, [trail], False, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.polylines(img, [trail], False, rgb(COLORS['expert']), 3, cv2.LINE_AA)
            if expert['frame'].iloc[-1] >= frame:
                cv2.circle(img, px(past[-1]), 8, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, px(past[-1]), 6, rgb(COLORS['expert']), -1, cv2.LINE_AA)
        label = f't = {frame / FPS:.1f} s  ({GIF_SPEED}x speed)'
        cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        images.append(np.vstack([img, strip]))

    # One palette for all frames (the map colors plus the drawing colors) keeps the background from flickering
    palette = Image.fromarray(images[0]).quantize(colors=240).getpalette()[:240 * 3]
    palette += [0] * (240 * 3 - len(palette))
    for color in list(COLORS.values()) + ['#FFFFFF', '#000000', '#282828']:
        palette += rgb(color)
    palette += [0] * (768 - len(palette))
    palette_image = Image.new('P', (1, 1))
    palette_image.putpalette(palette)
    frames = [Image.fromarray(img).quantize(palette=palette_image, dither=0) for img in images]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // GIF_FPS, loop=0)


def main():
    scenes = sorted(f[:-len('_meta.json')] for f in os.listdir(os.path.join(MAPS, 'metadata'))
                    if f.endswith('_meta.json'))
    parser = argparse.ArgumentParser(description='Generate one scene from a map and draw its input and output.')
    parser.add_argument('--scene', default='carla_town10_049', choices=scenes, help='map to use')
    parser.add_argument('--example', action='store_true',
                        help='skip generation and draw the example output in examples/topo_ft/')
    parser.add_argument('--force', action='store_true', help='generate the scene again even if its CSV exists')
    args = parser.parse_args()

    out_dir = os.path.join(ROOT, 'output', 'demo', args.scene)
    os.makedirs(out_dir, exist_ok=True)
    bg, m_per_px = show_input(args.scene, out_dir)
    if args.example:
        stem = os.path.join(EXAMPLES, f'{args.scene}-crowdes-synthetic-0')
        if not os.path.exists(stem + '.csv'):
            sys.exit(f'No example output for {args.scene} in {rel(EXAMPLES)}/.')
        print(f'\n[2/3] Generate  (skipped with --example, using {rel(EXAMPLES)}/)')
    else:
        stem = generate(args.scene, args.force, out_dir)
    show_output(args.scene, stem, bg, m_per_px, out_dir)
    print(f'\nDone. Open the pictures in {rel(out_dir)}/')


if __name__ == '__main__':
    main()
