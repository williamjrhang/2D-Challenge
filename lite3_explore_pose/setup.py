from glob import glob

from setuptools import find_packages, setup

package_name = 'lite3_explore_pose'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.rviz') + glob('config/*.xml')),
        ('share/' + package_name + '/config/bt', glob('config/bt/*.xml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Group 5',
    maintainer_email='todo@todo.todo',
    description='Arena coverage exploration for the Lite3 using SLAM and Nav2.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'explorer = lite3_explore_pose.explorer_node:main',
            'turn_hold = lite3_explore_pose.turn_hold:main',
             'mission_controller = lite3_explore_pose.mission_controller:main',
            'special_obstacles = lite3_explore_pose.special_obstacles:main',
            'vlm_watcher = lite3_explore_pose.vlm_watcher:main',
            'qr_reader = lite3_explore_pose.qr_reader:main',
            'reporter = lite3_explore_pose.reporter:main',
            'obstacle_detector = lite3_explore_pose.obstacle_detector:main',
            'link_relay = lite3_explore_pose.link_relay:main',
        ],
    },
)
