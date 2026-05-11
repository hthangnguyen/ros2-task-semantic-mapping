#!/usr/bin/env python3
"""
task_query_node.py
==================
Interactive terminal node. Lets the operator type a new task at runtime;
publishes it to /task/query so the SceneGraphNode updates relevance scores.

Also automatically cycles through tasks every N seconds for demo/GIF recording.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import threading
import time

DEMO_TASKS = [
    "pick up the mug",
    "work at the computer",
    "find emergency equipment",
    "prepare a presentation",
    "leave the room",
    "water the plants",
    "pack my belongings",
]

AUTO_CYCLE_SECONDS = 6.0   # change task every N seconds in auto-demo mode


class TaskQueryNode(Node):

    def __init__(self):
        super().__init__('task_query_node')

        self.declare_parameter('auto_demo', True)
        self.auto_demo = self.get_parameter('auto_demo').value

        self.pub = self.create_publisher(String, '/task/query', 10)
        self.task_idx = 0

        if self.auto_demo:
            self.get_logger().info(
                '[TaskQuery] AUTO DEMO mode — cycling tasks every '
                f'{AUTO_CYCLE_SECONDS}s. Set auto_demo:=false for manual mode.'
            )
            self._publish_task(DEMO_TASKS[self.task_idx])
            self.timer = self.create_timer(AUTO_CYCLE_SECONDS, self._cycle_task)
        else:
            self.get_logger().info('[TaskQuery] MANUAL mode — type a task and press Enter.')
            t = threading.Thread(target=self._input_loop, daemon=True)
            t.start()

    def _publish_task(self, task: str):
        msg = String()
        msg.data = task
        self.pub.publish(msg)
        self.get_logger().info(f'[TaskQuery] Published task → "{task}"')

    def _cycle_task(self):
        self.task_idx = (self.task_idx + 1) % len(DEMO_TASKS)
        self._publish_task(DEMO_TASKS[self.task_idx])

    def _input_loop(self):
        print("\n" + "="*60)
        print("  Task-Driven Scene Graph — interactive task input")
        print("="*60)
        print("Available tasks:")
        for i, t in enumerate(DEMO_TASKS):
            print(f"  [{i}] {t}")
        print("\nType a task number, or a free-form task string:")

        while True:
            try:
                raw = input(">> ").strip()
                if raw.isdigit() and int(raw) < len(DEMO_TASKS):
                    task = DEMO_TASKS[int(raw)]
                elif raw:
                    task = raw
                else:
                    continue
                self._publish_task(task)
            except (EOFError, KeyboardInterrupt):
                break


def main(args=None):
    rclpy.init(args=args)
    node = TaskQueryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()