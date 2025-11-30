from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mook_cinematic_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mobile',
    maintainer_email='mobile@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'path_player = mook_cinematic_navigation.path_player_node:main',
            'preview_renderer = mook_cinematic_navigation.render_node:main',
            'ply_to_occupancy = mook_cinematic_navigation.ply_to_occupancy_node:main',
            'cinematic_nav2_interactive = mook_cinematic_navigation.cinematic_nav2_interactive:main',
        ],
    },
)
