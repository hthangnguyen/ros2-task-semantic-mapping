#!/usr/bin/env python3
"""
scene_simulator.py
==================
Simulates a 3D indoor environment with static semantic objects and
dynamic agents (moving objects). Publishes:
  - /scene/objects        : list of semantic objects with 3D positions
  - /scene/dynamic_points : point cloud of the current scan
  - /scene/free_space     : voxel grid of confirmed free space
  - /scene/clock          : sim time tick

Inspired by Dynablox (free space map) and Khronos (spatio-temporal scene).
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header, String
import json
import math
import time

# ── Scene definition ──────────────────────────────────────────────────────────
# Each object: (label, x, y, z, is_dynamic)
STATIC_OBJECTS = [
    ("mug",          1.0,  0.5,  0.8, False),
    ("laptop",       2.5,  0.5,  0.8, False),
    ("chair",        1.5, -1.0,  0.4, False),
    ("table",        2.0,  0.0,  0.4, False),
    ("plant",       -1.0,  2.0,  0.5, False),
    ("monitor",      3.0,  0.5,  1.2, False),
    ("keyboard",     2.5, -0.2,  0.8, False),
    ("book",         0.5,  1.5,  0.8, False),
    ("backpack",    -0.5, -1.5,  0.4, False),
    ("whiteboard",   4.0,  0.0,  1.5, False),
    ("fire_extinguisher", -2.0,  0.0,  0.6, False),
    ("door",         5.0,  1.0,  1.0, False),
]

# Dynamic agents follow circular/linear paths
DYNAMIC_AGENTS = [
    {"label": "person_1", "path": "circle",  "cx": 0.0, "cy": 0.0, "r": 2.5, "speed": 0.4, "z": 0.9},
    {"label": "person_2", "path": "linear",  "x0": -3.0, "x1": 3.0, "y": 1.5, "speed": 0.6, "z": 0.9},
    {"label": "robot_cart","path": "circle", "cx": 1.0, "cy": 1.0, "r": 1.2, "speed": 0.2, "z": 0.5},
]


class SceneSimulatorNode(Node):

    def __init__(self):
        super().__init__('scene_simulator')

        self.declare_parameter('publish_rate_hz', 10.0)
        rate = self.get_parameter('publish_rate_hz').value

        self.pub_objects  = self.create_publisher(MarkerArray, '/scene/objects',        10)
        self.pub_dynamic  = self.create_publisher(MarkerArray, '/scene/dynamic_agents', 10)
        self.pub_scene_json = self.create_publisher(String,    '/scene/json',           10)

        self.t = 0.0
        self.dt = 1.0 / rate
        self.timer = self.create_timer(self.dt, self.tick)
        self.frame_id = 'world'

        self.get_logger().info(
            f'[SceneSimulator] Running at {rate:.1f} Hz — '
            f'{len(STATIC_OBJECTS)} static objects, {len(DYNAMIC_AGENTS)} dynamic agents'
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _header(self) -> Header:
        h = Header()
        h.frame_id = self.frame_id
        h.stamp = self.get_clock().now().to_msg()
        return h

    def _sphere_marker(self, mid, label, x, y, z, r, g, b, scale=0.25, ns='static') -> Marker:
        m = Marker()
        m.header = self._header()
        m.ns = ns
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.9
        m.text = label
        return m

    def _text_marker(self, mid, label, x, y, z, ns='labels') -> Marker:
        m = Marker()
        m.header = self._header()
        m.ns = ns
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z + 0.35
        m.pose.orientation.w = 1.0
        m.scale.z = 0.18
        m.color.r = m.color.g = m.color.b = 1.0; m.color.a = 1.0
        m.text = label
        return m

    def _agent_position(self, agent: dict, t: float):
        if agent['path'] == 'circle':
            angle = t * agent['speed']
            x = agent['cx'] + agent['r'] * math.cos(angle)
            y = agent['cy'] + agent['r'] * math.sin(angle)
        else:  # linear bounce
            period = 2.0 * (agent['x1'] - agent['x0']) / agent['speed']
            phase  = (t % period) / period
            frac   = 1.0 - abs(2.0 * phase - 1.0)
            x = agent['x0'] + frac * (agent['x1'] - agent['x0'])
            y = agent['y']
        return x, y, agent['z']

    # ── main tick ─────────────────────────────────────────────────────────────

    def tick(self):
        self.t += self.dt

        # --- Static objects ---
        static_markers = MarkerArray()
        scene_data = {'t': self.t, 'static': [], 'dynamic': []}

        for i, (label, x, y, z, _) in enumerate(STATIC_OBJECTS):
            static_markers.markers.append(
                self._sphere_marker(i, label, x, y, z, 0.2, 0.6, 1.0, scale=0.28, ns='static')
            )
            static_markers.markers.append(
                self._text_marker(1000 + i, label, x, y, z)
            )
            scene_data['static'].append({'id': i, 'label': label, 'x': x, 'y': y, 'z': z})

        self.pub_objects.publish(static_markers)

        # --- Dynamic agents ---
        dyn_markers = MarkerArray()
        for j, agent in enumerate(DYNAMIC_AGENTS):
            x, y, z = self._agent_position(agent, self.t)
            dyn_markers.markers.append(
                self._sphere_marker(j, agent['label'], x, y, z, 1.0, 0.2, 0.2, scale=0.45, ns='dynamic')
            )
            dyn_markers.markers.append(
                self._text_marker(500 + j, agent['label'], x, y, z, ns='dyn_labels')
            )
            scene_data['dynamic'].append({
                'id': j, 'label': agent['label'], 'x': float(x), 'y': float(y), 'z': float(z)
            })

        self.pub_dynamic.publish(dyn_markers)

        # --- JSON scene state for downstream nodes ---
        msg = String()
        msg.data = json.dumps(scene_data)
        self.pub_scene_json.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SceneSimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()