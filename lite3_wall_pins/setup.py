from glob import glob

from setuptools import find_packages, setup

package_name = 'lite3_wall_pins'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.rviz')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Group 5',
    maintainer_email='todo@todo.todo',
    description='Observe-only QR / picture pins on the SLAM map, shown in RViz.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wall_pins = lite3_wall_pins.pin_node:main',
        ],
    },
)
