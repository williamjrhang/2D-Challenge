from setuptools import setup

package_name = 'lite3_camera_recorder'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Group 5',
    maintainer_email='todo@todo.todo',
    description='Record the Lite3 obstacle camera to video and export frames for labeling.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'record_camera = lite3_camera_recorder.recorder:main',
            'extract_frames = lite3_camera_recorder.extract_frames:main',
        ],
    },
)
