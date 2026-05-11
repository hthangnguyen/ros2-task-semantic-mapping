from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'task_scene_graph'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='PhD Researcher',
    maintainer_email='you@example.com',
    description='Task-Driven Semantic Scene Graph with Dynamic Object Detection',
    license='MIT',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scene_simulator = task_scene_graph.scene_simulator:main',
            'dynamic_detector = task_scene_graph.dynamic_detector:main',
            'scene_graph_node = task_scene_graph.scene_graph_node:main',
            'task_query_node = task_scene_graph.task_query_node:main',
        ],
    },
)