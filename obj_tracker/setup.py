from setuptools import setup
from setuptools import find_packages

package_name = 'obj_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),  # Automatically add all packages
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adrian',
    maintainer_email='sochania@mcmaster.ca',
    description='2D Object tracking on axis aligned or oriented bounding box data.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_tracker = obj_tracker.obj_tracker:main',
            'tracker_bbox_viz = obj_tracker.tracker_viz_node:main'
        ],
    },
)
