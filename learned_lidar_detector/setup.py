from glob import glob
import os
from setuptools import find_packages
from setuptools import setup

package_name = 'learned_lidar_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name+'/configs'], #find_packages(include=[package_name]),  # Automatically add all packages
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name),
         glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name),
         glob('weights/*.*')),
        (os.path.join('share', package_name),
         glob('configs/*.*')),
        (os.path.join('share', package_name),
         ['rviz.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adrian',
    maintainer_email='adrian-soch@github.com',
    description='3D LiDAR Object detector using YOLOv8-obb and psuedo images',
    license='AL2',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'learned_lidar_detector = learned_lidar_detector.detection_node:main'
        ],
    },
)
