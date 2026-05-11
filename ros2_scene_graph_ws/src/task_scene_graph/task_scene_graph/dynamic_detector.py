#!/usr/bin/env python3
"""
dynamic_detector.py
===================
Implements the core Dynablox insight in 2D voxel space (simulation):

  "A point is dynamic if it currently occupies a voxel that was
   previously confirmed as FREE SPACE."

Free space is confirmed after a voxel has been observed empty for
T_w consecutive timesteps (with all neighbours also empty).

Subscribes:
  /scene/json     : full scene state (static + dynamic agent positions)

Publishes:
  /detection/dynamic_markers  : MarkerArray — red spheres for dynamic detections
  /detection/freespace_grid   : MarkerArray — grey cubes showing free space map
  /detection/stats_json       : detection statistics as JSON string
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Header
import json
import numpy as np
from collections import defaultdict

VOXEL_SIZE   = 0.5    # metres per voxel cell
T_w          = 8      # frames a voxel must be empty to become "free"
GRID_EXTENT  = 7.0    # ±7 m around origin
NEIGHBOUR_D  = 1      # neighbourhood radius in voxels


def world_to_voxel(x, y, z):
    """Convert world coords → integer voxel key."""
    ix = int(np.floor(x / VOXEL_SIZE))
    iy = int(np.floor(y / VOXEL_SIZE))
    iz = int(np.floor(z / VOXEL_SIZE))
    return (ix, iy, iz)


def voxel_center(vk):
    return (vk[0] * VOXEL_SIZE + VOXEL_SIZE / 2,
            vk[1] * VOXEL_SIZE + VOXEL_SIZE / 2,
            vk[2] * VOXEL_SIZE + VOXEL_SIZE / 2)


def neighbours(vk):
    ix, iy, iz = vk
    for dx in range(-NEIGHBOUR_D, NEIGHBOUR_D + 1):
        for dy in range(-NEIGHBOUR_D, NEIGHBOUR_D + 1):
            yield (ix + dx, iy + dy, iz)


class DynamicDetectorNode(Node):

    def __init__(self):
        super().__init__('dynamic_detector')

        # ── Free space map: voxel → frames_empty_count ────────────────────
        self.empty_count: dict = defaultdict(int)   # frames observed empty
        self.confirmed_free: set = set()             # voxels certified as free

        # ── Publisher ──────────────────────────────────────────────────────
        self.pub_dyn   = self.create_publisher(MarkerArray, '/detection/dynamic_markers', 10)
        self.pub_free  = self.create_publisher(MarkerArray, '/detection/freespace_grid',  10)
        self.pub_stats = self.create_publisher(String,      '/detection/stats_json',      10)

        # ── Subscriber ────────────────────────────────────────────────────
        self.sub = self.create_subscription(String, '/scene/json', self.on_scene, 10)

        self.frame_id = 'world'
        self.frame_count = 0
        self.total_detections = 0

        self.get_logger().info(
            f'[DynamicDetector] T_w={T_w} frames, voxel_size={VOXEL_SIZE}m  — '
            'waiting for /scene/json ...'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def on_scene(self, msg: String):
        scene = json.loads(msg.data)
        self.frame_count += 1

        # Collect all currently occupied voxels (both static + dynamic)
        occupied_voxels: set = set()

        for obj in scene.get('static', []):
            occupied_voxels.add(world_to_voxel(obj['x'], obj['y'], obj['z']))

        dynamic_voxels: set = set()
        for agent in scene.get('dynamic', []):
            vk = world_to_voxel(agent['x'], agent['y'], agent['z'])
            occupied_voxels.add(vk)
            dynamic_voxels.add(vk)

        # ── Step 1: update free-space map ─────────────────────────────────
        # For every grid cell in our extent, decide if it was empty this frame
        extent_voxels = int(GRID_EXTENT / VOXEL_SIZE)
        for ix in range(-extent_voxels, extent_voxels):
            for iy in range(-extent_voxels, extent_voxels):
                vk = (ix, iy, 1)  # ground floor layer
                if vk in occupied_voxels:
                    # reset empty counter — this voxel is occupied
                    self.empty_count[vk] = 0
                    self.confirmed_free.discard(vk)
                else:
                    self.empty_count[vk] += 1
                    # Confirm free after T_w frames of all neighbours also empty
                    if self.empty_count[vk] >= T_w:
                        if all(self.empty_count.get(n, 0) >= T_w for n in neighbours(vk)):
                            self.confirmed_free.add(vk)

        # ── Step 2: detect dynamics — Dynablox core rule ──────────────────
        # "current scan point in previously confirmed free space → DYNAMIC"
        detections = []
        for agent in scene.get('dynamic', []):
            vk = world_to_voxel(agent['x'], agent['y'], agent['z'])
            is_dyn_detected = any(n in self.confirmed_free for n in neighbours(vk))
            if is_dyn_detected:
                detections.append({
                    'label': agent['label'],
                    'x': agent['x'], 'y': agent['y'], 'z': agent['z'],
                    'voxel': vk,
                })
                self.total_detections += 1

        # ── Publish ───────────────────────────────────────────────────────
        self._publish_detections(detections)
        self._publish_free_space()
        self._publish_stats(detections)

    def _header(self):
        h = Header()
        h.frame_id = self.frame_id
        h.stamp = self.get_clock().now().to_msg()
        return h

    def _publish_detections(self, detections):
        ma = MarkerArray()
        for i, d in enumerate(detections):
            # Pulsing red sphere
            m = Marker()
            m.header = self._header()
            m.ns = 'dyndetect'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = d['x']
            m.pose.position.y = d['y']
            m.pose.position.z = d['z']
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.6
            m.color.r = 1.0; m.color.g = 0.15; m.color.b = 0.0; m.color.a = 0.85
            ma.markers.append(m)

            # "DYNAMIC" text label
            t = Marker()
            t.header = self._header()
            t.ns = 'dynlabels'
            t.id = 100 + i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = d['x']
            t.pose.position.y = d['y']
            t.pose.position.z = d['z'] + 0.55
            t.pose.orientation.w = 1.0
            t.scale.z = 0.20
            t.color.r = 1.0; t.color.g = 1.0; t.color.b = 0.0; t.color.a = 1.0
            t.text = f"⚠ DYNAMIC: {d['label']}"
            ma.markers.append(t)

        self.pub_dyn.publish(ma)

    def _publish_free_space(self):
        """Visualise confirmed free voxels as semi-transparent grey cubes."""
        ma = MarkerArray()
        # Only show a random subset for performance
        sample = list(self.confirmed_free)
        if len(sample) > 500:
            import random
            sample = random.sample(sample, 500)

        for i, vk in enumerate(sample):
            cx, cy, cz = voxel_center(vk)
            m = Marker()
            m.header = self._header()
            m.ns = 'freespace'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = cx
            m.pose.position.y = cy
            m.pose.position.z = cz
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = VOXEL_SIZE * 0.92
            m.color.r = 0.4; m.color.g = 0.8; m.color.b = 0.4; m.color.a = 0.08
            ma.markers.append(m)

        self.pub_free.publish(ma)

    def _publish_stats(self, detections):
        stats = {
            'frame': self.frame_count,
            'confirmed_free_voxels': len(self.confirmed_free),
            'dynamic_detections_this_frame': len(detections),
            'cumulative_detections': self.total_detections,
            'labels': [d['label'] for d in detections],
        }
        msg = String()
        msg.data = json.dumps(stats)
        self.pub_stats.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()