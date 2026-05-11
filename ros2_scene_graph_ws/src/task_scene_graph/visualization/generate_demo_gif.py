#!/usr/bin/env python3
"""
visualization/generate_demo_gif.py
===================================
Standalone script — generates an animated GIF demonstrating the
Task-Driven Semantic Scene Graph system.

Does NOT require ROS2 to run. Reproduces the same logic as the
ROS2 nodes (SceneSimulator + DynamicDetector + SceneGraphNode)
purely in Python + matplotlib.

Usage:
  pip install matplotlib numpy pillow
  python3 generate_demo_gif.py

Output:
  task_scene_graph_demo.gif   (~80 frames, 7 tasks)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import math
import io
from PIL import Image

# ── Colour theme ──────────────────────────────────────────────────────────────
BG        = '#0d1117'
GRID_COL  = '#21262d'
TEXT_COL  = '#e6edf3'
ACCENT    = '#58a6ff'
DYN_RED   = '#ff4444'
FREE_COL  = '#1a3a1a'

# ── Scene definition ──────────────────────────────────────────────────────────
STATIC_OBJECTS = [
    ("mug",          1.0,  0.5),
    ("laptop",       2.5,  0.5),
    ("chair",        1.5, -1.0),
    ("table",        2.0,  0.0),
    ("plant",       -1.0,  2.0),
    ("monitor",      3.0,  0.5),
    ("keyboard",     2.5, -0.2),
    ("book",         0.5,  1.5),
    ("backpack",    -0.5, -1.5),
    ("whiteboard",   4.0,  0.0),
    ("fire_ext",    -2.0,  0.0),
    ("door",         5.0,  1.0),
]

DYNAMIC_AGENTS = [
    {"label": "person_1", "path": "circle",  "cx": 0.0, "cy": 0.0, "r": 2.5, "speed": 0.35},
    {"label": "person_2", "path": "linear",  "x0": -3.0, "x1": 3.0, "y": 1.5, "speed": 0.5},
    {"label": "cart",     "path": "circle",  "cx": 1.0, "cy": 1.0, "r": 1.2, "speed": 0.2},
]

OBJECT_EMBEDDINGS = {
    "mug":        np.array([0.9, 0.1, 0.1, 0.0, 0.0, 0.8, 0.3]),
    "laptop":     np.array([0.6, 0.1, 0.9, 0.0, 0.0, 0.0, 0.4]),
    "chair":      np.array([0.2, 0.9, 0.0, 0.0, 0.1, 0.0, 0.1]),
    "table":      np.array([0.3, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "plant":      np.array([0.2, 0.2, 0.0, 0.0, 0.9, 0.1, 0.2]),
    "monitor":    np.array([0.3, 0.1, 0.9, 0.0, 0.0, 0.0, 0.3]),
    "keyboard":   np.array([0.7, 0.0, 0.8, 0.0, 0.0, 0.0, 0.3]),
    "book":       np.array([0.7, 0.1, 0.3, 0.0, 0.0, 0.0, 0.5]),
    "backpack":   np.array([0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.9]),
    "whiteboard": np.array([0.2, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0]),
    "fire_ext":   np.array([0.5, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "door":       np.array([0.1, 0.5, 0.0, 0.2, 0.0, 0.0, 0.0]),
}

TASKS = [
    ("pick up the mug",          np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.8, 0.3])),
    ("work at the computer",     np.array([0.3, 0.1, 0.9, 0.0, 0.0, 0.0, 0.3])),
    ("find emergency equipment", np.array([0.4, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0])),
    ("prepare a presentation",   np.array([0.3, 0.5, 0.6, 0.0, 0.0, 0.0, 0.2])),
    ("leave the room",           np.array([0.0, 0.3, 0.0, 0.1, 0.0, 0.0, 0.5])),
    ("water the plants",         np.array([0.5, 0.0, 0.0, 0.0, 0.9, 0.3, 0.0])),
    ("pack my belongings",       np.array([0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.9])),
]
ALPHA = 0.35   # null-task threshold


def cosine_sim(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-9 else 0.0


def agent_pos(agent, t):
    if agent['path'] == 'circle':
        a = t * agent['speed']
        return agent['cx'] + agent['r'] * math.cos(a), agent['cy'] + agent['r'] * math.sin(a)
    period = 2.0 * (agent['x1'] - agent['x0']) / agent['speed']
    phase  = (t % period) / period
    frac   = 1.0 - abs(2.0 * phase - 1.0)
    return agent['x0'] + frac * (agent['x1'] - agent['x0']), agent['y']


def relevance_color(score, irrelevant):
    if irrelevant:
        return (0.35, 0.35, 0.40, 0.6)
    if score < 0.5:
        t = score / 0.5
        return (0.0, 0.3 + t * 0.5, 1.0 - t * 0.5, 0.92)
    t = (score - 0.5) / 0.5
    return (0.3 + t * 0.7, 0.8 - t * 0.8, 0.0, 0.92)


def score_objects(task_emb):
    results = []
    for label, x, y in STATIC_OBJECTS:
        emb = OBJECT_EMBEDDINGS.get(label, np.zeros(7))
        sim = cosine_sim(emb, task_emb)
        results.append({'label': label, 'x': x, 'y': y, 'score': sim,
                        'irrelevant': sim < ALPHA})
    return results


# ── Free-space voxel tracker (Dynablox logic, 2D) ────────────────────────────
class FreeSpaceTracker:
    VSIZE = 0.5

    def __init__(self):
        self.empty_count = {}
        self.confirmed   = set()

    def _vk(self, x, y):
        return (int(math.floor(x / self.VSIZE)), int(math.floor(y / self.VSIZE)))

    def update(self, agent_positions):
        occupied = set(self._vk(x, y) for x, y in agent_positions)
        ext = int(7 / self.VSIZE)
        for ix in range(-ext, ext):
            for iy in range(-ext, ext):
                vk = (ix, iy)
                if vk in occupied:
                    self.empty_count[vk] = 0
                    self.confirmed.discard(vk)
                else:
                    self.empty_count[vk] = self.empty_count.get(vk, 0) + 1
                    if self.empty_count[vk] >= 8:
                        nbrs = [(ix + dx, iy + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
                        if all(self.empty_count.get(n, 0) >= 8 for n in nbrs):
                            self.confirmed.add(vk)

    def is_detected(self, x, y):
        vk = self._vk(x, y)
        nbrs = [(vk[0] + dx, vk[1] + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
        return any(n in self.confirmed for n in nbrs)


# ── Render one frame ──────────────────────────────────────────────────────────
def render_frame(t, task_name, task_emb, fst: FreeSpaceTracker):
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1], wspace=0.05)
    ax  = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1])

    ax.set_facecolor(BG)
    axr.set_facecolor(BG)

    # ── Grid ──────────────────────────────────────────────────────────────
    for gx in np.arange(-3.5, 6.5, 1.0):
        ax.axvline(gx, color=GRID_COL, lw=0.5, zorder=0)
    for gy in np.arange(-2.5, 3.5, 1.0):
        ax.axhline(gy, color=GRID_COL, lw=0.5, zorder=0)

    # ── Free-space voxels ─────────────────────────────────────────────────
    vs = fst.VSIZE
    for (ix, iy) in list(fst.confirmed)[:800]:
        cx, cy = ix * vs + vs / 2, iy * vs + vs / 2
        if -4 < cx < 7 and -3 < cy < 4:
            rect = plt.Rectangle((cx - vs/2, cy - vs/2), vs, vs,
                                  color='#1e4d1e', alpha=0.18, zorder=1)
            ax.add_patch(rect)

    # ── Score objects ──────────────────────────────────────────────────────
    scored = score_objects(task_emb)
    relevant = [o for o in scored if not o['irrelevant']]

    # Draw edges between relevant nodes
    for i in range(len(relevant)):
        for j in range(i + 1, len(relevant)):
            a, b = relevant[i], relevant[j]
            ax.plot([a['x'], b['x']], [a['y'], b['y']],
                    color='#3a6a8a', lw=0.8, alpha=0.5, zorder=2, linestyle='--')

    # Draw nodes
    for obj in scored:
        col = relevance_color(obj['score'], obj['irrelevant'])
        r_viz = 0.12 + (0 if obj['irrelevant'] else obj['score'] * 0.25)
        circle = plt.Circle((obj['x'], obj['y']), r_viz, color=col, zorder=3)
        ax.add_patch(circle)

        # Label
        score_str = "" if obj['irrelevant'] else f" {obj['score']:.2f}"
        prefix = "" if obj['irrelevant'] else "★ "
        txt_col = TEXT_COL if not obj['irrelevant'] else '#666677'
        ax.text(obj['x'], obj['y'] + r_viz + 0.18,
                f"{prefix}{obj['label']}{score_str}",
                ha='center', va='bottom', fontsize=6.5,
                color=txt_col, fontweight='bold' if not obj['irrelevant'] else 'normal',
                path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)],
                zorder=5)

    # ── Dynamic agents ────────────────────────────────────────────────────
    agent_positions = []
    for agent in DYNAMIC_AGENTS:
        x, y = agent_pos(agent, t)
        agent_positions.append((x, y))

        detected = fst.is_detected(x, y)
        edge_col = DYN_RED if detected else '#ff8844'
        pulse = 0.22 + 0.07 * math.sin(t * 3)

        outer = plt.Circle((x, y), pulse + 0.06,
                             color=edge_col, alpha=0.3, zorder=4)
        inner = plt.Circle((x, y), pulse,
                             color=DYN_RED if detected else '#ff6633',
                             alpha=0.9, zorder=5)
        ax.add_patch(outer)
        ax.add_patch(inner)

        label = f"⚠ {agent['label']}" if detected else agent['label']
        ax.text(x, y + pulse + 0.22, label,
                ha='center', va='bottom', fontsize=6.5,
                color=DYN_RED if detected else '#ffaa88', fontweight='bold',
                path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)],
                zorder=6)

    fst.update(agent_positions)

    # ── Main panel decorations ────────────────────────────────────────────
    ax.set_xlim(-3.8, 6.5)
    ax.set_ylim(-2.8, 3.5)
    ax.set_aspect('equal')
    ax.tick_params(colors=TEXT_COL, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)

    ax.set_title('3D Scene View  (RViz2)', color=TEXT_COL, fontsize=9,
                 fontfamily='monospace', pad=6)
    ax.set_xlabel('X (m)', color=TEXT_COL, fontsize=7)
    ax.set_ylabel('Y (m)', color=TEXT_COL, fontsize=7)

    # Task label at top
    fig.text(0.5, 0.97,
             f'Task  →  "{task_name}"',
             ha='center', va='top', fontsize=11, color=ACCENT,
             fontfamily='monospace', fontweight='bold')

    # ── Right panel: relevance bar chart ─────────────────────────────────
    labels  = [o['label'] for o in scored]
    scores  = [o['score'] for o in scored]
    colors  = [relevance_color(o['score'], o['irrelevant']) for o in scored]

    y_pos = np.arange(len(labels))
    bars  = axr.barh(y_pos, scores, color=colors, height=0.65, zorder=2)
    axr.axvline(ALPHA, color='#ff6633', lw=1.2, linestyle='--', zorder=3, alpha=0.9)
    axr.text(ALPHA + 0.01, len(labels) - 0.3, 'θ (threshold)',
             color='#ff6633', fontsize=6.5, va='top', fontfamily='monospace')

    axr.set_yticks(y_pos)
    axr.set_yticklabels(labels, color=TEXT_COL, fontsize=7.5)
    axr.set_xlim(0, 1.05)
    axr.set_xlabel('Task Relevance Score', color=TEXT_COL, fontsize=7.5)
    axr.set_title('Clio Scoring', color=TEXT_COL, fontsize=9, fontfamily='monospace', pad=6)
    axr.tick_params(colors=TEXT_COL, labelsize=7)
    axr.set_facecolor(BG)
    for spine in axr.spines.values():
        spine.set_edgecolor(GRID_COL)
    axr.grid(axis='x', color=GRID_COL, lw=0.5)

    # Legend
    handles = [
        mpatches.Patch(color='#4488ff', label='Task Relevant'),
        mpatches.Patch(color='#555566', label='Irrelevant'),
        mpatches.Patch(color=DYN_RED,   label='Dynamic (detected)'),
        mpatches.Patch(color='#1e4d1e', label='Free Space (Dynablox)'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=6.5,
              facecolor='#1a1f26', edgecolor=GRID_COL,
              labelcolor=TEXT_COL, framealpha=0.85)

    # ── Paper attribution (corner watermark) ──────────────────────────────
    fig.text(0.01, 0.01,
             'Inspired by: Clio (IB + CLIP) · Dynablox (free-space violation) · Khronos (scene graph)',
             fontsize=5.5, color='#444466', va='bottom', fontfamily='monospace')

    fig.text(0.99, 0.01, f't = {t:.1f}s',
             fontsize=6, color='#333355', ha='right', va='bottom', fontfamily='monospace')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Generating task_scene_graph_demo.gif ...")
    print("(~80 frames across 7 tasks — may take ~30 seconds)")

    fst    = FreeSpaceTracker()
    frames = []
    fps    = 12
    task_duration = 6.0   # seconds per task
    dt     = 1.0 / fps

    t = 0.0
    for task_name, task_emb in TASKS:
        print(f"  Rendering task: '{task_name}' ...")
        task_t_end = t + task_duration
        while t < task_t_end:
            frame = render_frame(t, task_name, task_emb, fst)
            frames.append(frame)
            t += dt

    # Save GIF
    out_path = 'task_scene_graph_demo.gif'
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )
    print(f"\n✓ Saved → {out_path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")


if __name__ == '__main__':
    main()