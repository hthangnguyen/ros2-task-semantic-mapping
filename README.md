# Task-Driven Semantic Scene Graph with Dynamic Object Detection

> A ROS2 project demonstrating integration of ideas from
> **Clio**, **Dynablox**, and **Khronos** — without any physical robot or sensor hardware.

![Demo GIF](ros2_scene_graph_ws/task_scene_graph_demo.gif)

---

## Overview

This project implements a **task-driven 3D semantic scene graph** running entirely in simulation on Ubuntu 22.04. Given a natural language task (e.g., *"pick up the mug"*), the system:

1. **Scores every object** in the scene by its task relevance using CLIP-style cosine similarity (→ Clio)
2. **Groups objects** into semantic clusters via an IB-inspired JS-divergence agglomerative merge (→ Clio)
3. **Detects dynamic agents** in real time by checking whether their current position violates previously confirmed free space (→ Dynablox)
4. **Publishes a hierarchical scene graph** with temporal state — relevant objects highlighted, irrelevant ones greyed out (→ Khronos)
5. **Visualises everything live in RViz2**, including free-space voxels, dynamic detections, and task-relevance heat-map

---

## Research Connections

| Paper | What this project implements |
|---|---|
| **Clio** (Maggio et al., 2024) | CLIP cosine similarity as task-relevance score; IB-inspired agglomerative clustering via JS-divergence; null-task threshold α |
| **Dynablox** (Schmid et al., 2023) | Free-space confirmation after T_w empty frames; neighbourhood check for robustness; dynamic detection as free-space violation |
| **Khronos** (Schmid et al., 2024) | Scene graph with per-object state history; local (active window) vs. global (reconciled) scene representation |

### Key formulas implemented

**Task-relevance score (Clio Eq. 4):**
```
θ(xᵢ)_j = cos(f_{xᵢ}, f_{t_j})   for task j
```

**IB merge criterion (Clio Eq. 2):**
```
d_ij = JS[ p(y|xᵢ), p(y|x_j) ]
```
Objects with `d_ij < 0.05` are merged into the same semantic cluster.

**Free-space label (Dynablox Eq. 5):**
```
f = 𝟙( t_occ(v') < t - T_w  ∧  w(v') > 0,  ∀v' ∈ 𝒩(v) )
```

**Dynamic detection (Dynablox Eq. 6):**
```
V_dyn = { v_k ∈ K(P^t) | ∃v' ∈ 𝒩(v_k) : f(v') = 1 }
```

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        ROS2 Node Graph                         │
│                                                                │
│  ┌──────────────────┐    /scene/json     ┌──────────────────┐  │
│  │  scene_simulator │ ─────────────────► │ dynamic_detector │  │
│  │                  │                    │                  │  │
│  │  • 12 static     │    /scene/objects  │ • Free-space map │  │
│  │    semantic objs │ ──────────────►    │ • Dynablox logic │  │
│  │  • 3 dynamic     │    /scene/dynamic  │ • T_w=8 frames   │  │
│  │    agents        │ ──────────────►    └──────────────────┘  │
│  └──────────────────┘                           │              │
│                    /detection/dynamic_markers   │              │
│  ┌──────────────────┐    /task/query            ▼              │
│  │  task_query_node │ ────────────► ┌──────────────────┐       │
│  │                  │               │  scene_graph_node│       │
│  │  • Auto-cycles   │  /scene/json  │                  │       │
│  │    7 tasks       │ ────────────► │ • CLIP scoring   │       │
│  │  • 6s per task   │               │ • IB clustering  │       │
│  └──────────────────┘               │ • Graph builder  │       │
│                                     └──────────────────┘       │
│                                              │                 │
│                     /scene_graph/nodes  /scene_graph/edges     │
│                                    ▼                           │
│                             ┌──────────────┐                   │
│                             │    RViz2     │                   │
│                             │ Visualizer   │                   │
│                             └──────────────┘                   │
└────────────────────────────────────────────────────────────────┘
```

---

## Topics Published

| Topic | Type | Description |
|---|---|---|
| `/scene/objects` | `MarkerArray` | Static objects (blue spheres) |
| `/scene/dynamic_agents` | `MarkerArray` | Moving agents (red spheres) |
| `/scene/json` | `String` | Full scene state as JSON |
| `/detection/dynamic_markers` | `MarkerArray` | Dynablox detections (bright red + label) |
| `/detection/freespace_grid` | `MarkerArray` | Confirmed free voxels (green cubes) |
| `/detection/stats_json` | `String` | Detection statistics |
| `/scene_graph/nodes` | `MarkerArray` | Scene graph nodes (heat-map coloured) |
| `/scene_graph/edges` | `Marker` | Edges between task-relevant nodes |
| `/scene_graph/json` | `String` | Full graph state as JSON |
| `/task/query` | `String` | Input: natural language task |

---

## Installation

### Requirements
- Ubuntu 22.04 (Jammy)
- ROS2 Humble
- Python 3.10+
- No GPU, sensors, or hardware required

### Setup

```bash
# 1. Clone / place this repo
cd ~/ros2_scene_graph_ws/src
# (copy task_scene_graph/ here)

# 2. Install Python dependencies
pip3 install numpy open3d --break-system-packages

# 3. Build
cd ~/ros2_scene_graph_ws
colcon build --packages-select task_scene_graph
source install/setup.bash
```

---

## Running

### Full system (auto demo — cycles tasks automatically)
```bash
ros2 launch task_scene_graph task_scene_graph.launch.py
```

### Open RViz2
```bash
rviz2 -d src/task_scene_graph/config/scene_graph.rviz
```

### Manual task input
```bash
ros2 launch task_scene_graph task_scene_graph.launch.py auto_demo:=false
# Then in another terminal:
ros2 topic pub /task/query std_msgs/String "data: 'find emergency equipment'" --once
```

### Run individual nodes
```bash
ros2 run task_scene_graph scene_simulator
ros2 run task_scene_graph dynamic_detector
ros2 run task_scene_graph scene_graph_node
ros2 run task_scene_graph task_query_node
```

### Generate demo GIF (no ROS2 needed)
```bash
pip3 install matplotlib pillow
python3 visualization/generate_demo_gif.py
# → task_scene_graph_demo.gif
```

---

## What the Visualization Shows

| Visual element | Meaning |
|---|---|
| **Blue→Yellow→Red spheres** | Objects scored by task relevance (blue=low, red=high) |
| **Grey spheres** | Task-irrelevant objects (score < θ = 0.35) |
| **Sphere size** | Proportional to relevance score |
| **Dashed edges** | Scene graph connections between task-relevant objects |
| **Bright red large spheres** | Dynablox-detected dynamic agents |
| **⚠ DYNAMIC label** | Agent detected in previously free space |
| **Translucent green voxels** | Confirmed free space map |
| **Right panel bar chart** | Per-object relevance scores with null-task threshold |

---



## This project demonstrates understanding of three active research areas:

**1. Task-conditioned scene representation** — The IB objective from Clio formalises something robotics researchers have long known intuitively: a useful map is *task-specific*, not complete. Implementing this from scratch shows command of information-theoretic principles.

**2. Geometry-only dynamic detection** — Dynablox's key insight (free-space violation) achieves state-of-the-art dynamic detection *without any learned appearance model*. This is a powerful result for real-world robots where training data is scarce.

**3. Unified spatio-temporal scene graphs** — Khronos's factorisation of the SMS problem is the theoretical backbone for next-generation robot memory systems. Understanding this architecture is prerequisite for contributing to the LP2 / DAAAM line of work.

**Possible extensions:**
- Replace the hand-crafted embeddings with a real CLIP encoder (one `pip install open_clip_torch`)
- Add LP2's CTMC-based human trajectory prediction on top of the dynamic detections
- Implement Khronos-style temporal reconciliation (object reappearance detection)
- Plug in a real depth camera (RealSense, ZED) — the node interfaces are already ROS2-compatible

---

## File Structure

```
task_scene_graph/
├── package.xml
├── setup.py
├── task_scene_graph/
│   ├── __init__.py
│   ├── scene_simulator.py      # Synthetic scene + dynamic agents
│   ├── dynamic_detector.py     # Dynablox free-space violation logic
│   ├── scene_graph_node.py     # Clio scoring + IB clustering + graph
│   └── task_query_node.py      # Interactive / auto task publisher
├── launch/
│   └── task_scene_graph.launch.py
├── config/
│   └── scene_graph.rviz
└── visualization/
    └── generate_demo_gif.py    # Standalone demo (no ROS2 required)
```