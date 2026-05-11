#!/usr/bin/env python3
"""
scene_graph_node.py
===================
Implements the core Clio insight in a pure-Python simulation:

  Given a natural language task, score each object in the scene by its
  CLIP-based cosine similarity to the task embedding. Objects above a
  threshold θ are "task-relevant"; the rest are irrelevant.

  The IB-inspired agglomerative merge is approximated by grouping
  objects with similar task-relevance profiles into scene graph clusters
  (rooms / functional zones).

Subscribes:
  /scene/json         : current scene state
  /task/query         : String — natural language task from operator

Publishes:
  /scene_graph/nodes      : MarkerArray — coloured by task relevance
  /scene_graph/edges      : Marker (LINE_LIST) — graph edges
  /scene_graph/json       : JSON representation of the current scene graph
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, ColorRGBA, Header
from geometry_msgs.msg import Point
import json
import numpy as np
import math

# ── CLIP-style cosine similarities (pre-computed for demo, no GPU needed) ────
# In a real system: embeddings = clip_model.encode(texts)
# Here we use a hand-crafted semantic embedding table that mirrors
# how CLIP places related concepts near each other.
# Dimensions represent semantic axes:
#   [graspable, furniture, digital, safety, nature, food/drink, personal]
OBJECT_EMBEDDINGS = {
    "mug":               np.array([0.9,  0.1,  0.1,  0.0,  0.0,  0.8,  0.3]),
    "laptop":            np.array([0.6,  0.1,  0.9,  0.0,  0.0,  0.0,  0.4]),
    "chair":             np.array([0.2,  0.9,  0.0,  0.0,  0.1,  0.0,  0.1]),
    "table":             np.array([0.3,  0.9,  0.0,  0.0,  0.0,  0.0,  0.0]),
    "plant":             np.array([0.2,  0.2,  0.0,  0.0,  0.9,  0.1,  0.2]),
    "monitor":           np.array([0.3,  0.1,  0.9,  0.0,  0.0,  0.0,  0.3]),
    "keyboard":          np.array([0.7,  0.0,  0.8,  0.0,  0.0,  0.0,  0.3]),
    "book":              np.array([0.7,  0.1,  0.3,  0.0,  0.0,  0.0,  0.5]),
    "backpack":          np.array([0.7,  0.2,  0.0,  0.0,  0.0,  0.0,  0.9]),
    "whiteboard":        np.array([0.2,  0.6,  0.4,  0.0,  0.0,  0.0,  0.0]),
    "fire_extinguisher": np.array([0.5,  0.0,  0.0,  0.9,  0.0,  0.0,  0.0]),
    "door":              np.array([0.1,  0.5,  0.0,  0.2,  0.0,  0.0,  0.0]),
}

TASK_EMBEDDINGS = {
    "pick up the mug":           np.array([0.9,  0.0,  0.0,  0.0,  0.0,  0.8,  0.3]),
    "work at the computer":      np.array([0.3,  0.1,  0.9,  0.0,  0.0,  0.0,  0.3]),
    "find emergency equipment":  np.array([0.4,  0.0,  0.0,  0.9,  0.0,  0.0,  0.0]),
    "prepare a presentation":    np.array([0.3,  0.5,  0.6,  0.0,  0.0,  0.0,  0.2]),
    "leave the room":            np.array([0.0,  0.3,  0.0,  0.1,  0.0,  0.0,  0.5]),
    "water the plants":          np.array([0.5,  0.0,  0.0,  0.0,  0.9,  0.3,  0.0]),
    "pack my belongings":        np.array([0.8,  0.0,  0.2,  0.0,  0.0,  0.0,  0.9]),
}

NULL_TASK_THRESHOLD = 0.35   # α — objects below this are task-irrelevant


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def relevance_to_rgb(score: float, is_irrelevant: bool):
    """Map score ∈ [0,1] → (R,G,B) heat map. Irrelevant → grey."""
    if is_irrelevant:
        return (0.45, 0.45, 0.45)
    # Blue (low) → Green → Yellow → Red (high)
    if score < 0.5:
        t = score / 0.5
        return (0.0, t, 1.0 - t)
    else:
        t = (score - 0.5) / 0.5
        return (t, 1.0 - t * 0.5, 0.0)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence — used in Clio's merge criterion."""
    p = p + 1e-12; q = q + 1e-12
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


class SceneGraphNode(Node):

    def __init__(self):
        super().__init__('scene_graph_node')

        self.current_task = "pick up the mug"   # default task
        self.task_embedding = TASK_EMBEDDINGS[self.current_task]

        # Publishers
        self.pub_nodes = self.create_publisher(MarkerArray, '/scene_graph/nodes', 10)
        self.pub_edges = self.create_publisher(Marker,      '/scene_graph/edges', 10)
        self.pub_json  = self.create_publisher(String,      '/scene_graph/json',  10)

        # Subscribers
        self.sub_scene = self.create_subscription(String, '/scene/json',   self.on_scene, 10)
        self.sub_task  = self.create_subscription(String, '/task/query',   self.on_task,  10)

        self.frame_id = 'world'
        self.get_logger().info(
            f'[SceneGraph] Clio-inspired scorer active — '
            f'default task: "{self.current_task}"'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def on_task(self, msg: String):
        task = msg.data.strip()
        if task in TASK_EMBEDDINGS:
            self.current_task = task
            self.task_embedding = TASK_EMBEDDINGS[task]
            self.get_logger().info(f'[SceneGraph] New task: "{task}"')
        else:
            # Compute similarity against all known tasks, pick closest
            best, best_sim = self.current_task, -1.0
            q = np.array([0.0] * 7)  # fallback zero embedding
            for known_task, emb in TASK_EMBEDDINGS.items():
                # simple word overlap heuristic for unknown tasks
                words = set(task.lower().split())
                known_words = set(known_task.lower().split())
                sim = len(words & known_words) / max(len(words | known_words), 1)
                if sim > best_sim:
                    best_sim = sim; best = known_task
            self.current_task = best
            self.task_embedding = TASK_EMBEDDINGS[best]
            self.get_logger().warn(
                f'[SceneGraph] Unknown task — using closest: "{best}" (sim={best_sim:.2f})'
            )

    def on_scene(self, msg: String):
        scene = json.loads(msg.data)
        objects = scene.get('static', [])

        # ── Step 1: compute per-object task-relevance (Clio θ vector) ────
        scored = []
        for obj in objects:
            label = obj['label']
            emb = OBJECT_EMBEDDINGS.get(label, np.zeros(7))
            sim = cosine_sim(emb, self.task_embedding)
            is_irrelevant = sim < NULL_TASK_THRESHOLD
            scored.append({**obj, 'relevance': sim, 'irrelevant': is_irrelevant})

        # ── Step 2: IB-inspired grouping by JS-divergence ─────────────────
        # Build conditional distributions p(y|x_i) over "task" dimension
        task_probs = {}
        for o in scored:
            label = o['label']
            emb = OBJECT_EMBEDDINGS.get(label, np.zeros(7))
            # p(y|x_i) is the softmax-normalised task embedding similarity
            raw = np.array([cosine_sim(emb, te) for te in TASK_EMBEDDINGS.values()])
            raw = np.clip(raw, 0, 1)
            task_probs[label] = raw / (raw.sum() + 1e-9)

        # Greedy merge: objects with JS < 0.05 are in the same cluster
        clusters = []
        assigned = {}
        for o in scored:
            label = o['label']
            placed = False
            for cid, cluster in enumerate(clusters):
                rep = cluster[0]['label']
                jsd = js_divergence(task_probs[label], task_probs[rep])
                if jsd < 0.05:
                    cluster.append(o)
                    assigned[label] = cid
                    placed = True
                    break
            if not placed:
                assigned[label] = len(clusters)
                clusters.append([o])

        # ── Step 3: publish node markers ──────────────────────────────────
        node_markers = MarkerArray()
        json_nodes = []

        for i, obj in enumerate(scored):
            r, g, b = relevance_to_rgb(obj['relevance'], obj['irrelevant'])

            # Sphere scaled by relevance
            scale = 0.18 + obj['relevance'] * 0.28 if not obj['irrelevant'] else 0.18
            m = Marker()
            m.header = self._header()
            m.ns = 'sg_nodes'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = obj['x']
            m.pose.position.y = obj['y']
            m.pose.position.z = obj['z']
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.95
            node_markers.markers.append(m)

            # Relevance label
            label_score = f"{obj['label']}\n{obj['relevance']:.2f}"
            if not obj['irrelevant']:
                label_score = f"★ {label_score}"
            t = Marker()
            t.header = self._header()
            t.ns = 'sg_labels'
            t.id = 500 + i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = obj['x']
            t.pose.position.y = obj['y']
            t.pose.position.z = obj['z'] + scale + 0.12
            t.pose.orientation.w = 1.0
            t.scale.z = 0.13
            t.color.r = r; t.color.g = g; t.color.b = b; t.color.a = 1.0
            t.text = label_score
            node_markers.markers.append(t)

            json_nodes.append({
                'id': i,
                'label': obj['label'],
                'x': obj['x'], 'y': obj['y'], 'z': obj['z'],
                'relevance': round(obj['relevance'], 3),
                'irrelevant': obj['irrelevant'],
                'cluster': assigned[obj['label']],
            })

        self.pub_nodes.publish(node_markers)

        # ── Step 4: publish edges between task-relevant nodes ─────────────
        edge_marker = Marker()
        edge_marker.header = self._header()
        edge_marker.ns = 'sg_edges'
        edge_marker.id = 0
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.025
        edge_marker.color.r = 0.7; edge_marker.color.g = 0.7
        edge_marker.color.b = 0.7; edge_marker.color.a = 0.4

        relevant = [o for o in scored if not o['irrelevant']]
        for i in range(len(relevant)):
            for j in range(i + 1, len(relevant)):
                a, b_ = relevant[i], relevant[j]
                p1 = Point(x=a['x'], y=a['y'], z=a['z'])
                p2 = Point(x=b_['x'], y=b_['y'], z=b_['z'])
                edge_marker.points.append(p1)
                edge_marker.points.append(p2)

        self.pub_edges.publish(edge_marker)

        # ── Step 5: publish JSON ───────────────────────────────────────────
        graph_json = {
            'task': self.current_task,
            'n_total': len(scored),
            'n_relevant': sum(1 for o in scored if not o['irrelevant']),
            'n_clusters': len(clusters),
            'nodes': json_nodes,
        }
        jmsg = String(); jmsg.data = json.dumps(graph_json)
        self.pub_json.publish(jmsg)

    def _header(self):
        h = Header()
        h.frame_id = self.frame_id
        h.stamp = self.get_clock().now().to_msg()
        return h


def main(args=None):
    rclpy.init(args=args)
    node = SceneGraphNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()