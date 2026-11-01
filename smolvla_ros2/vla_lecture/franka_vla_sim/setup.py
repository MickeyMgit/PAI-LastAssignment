from glob import glob

from setuptools import setup

package_name = 'franka_vla_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/urdf', glob('urdf/*.xacro')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maintainer',
    maintainer_email='maintainer@example.com',
    description='VLA data collection / inference / evaluation on Gazebo FR3',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'teleop_keyboard = franka_vla_sim.teleop_keyboard:main',
            'vla_client = franka_vla_sim.vla_client:main',
            'action_executor = franka_vla_sim.action_executor:main',
            'evaluator = franka_vla_sim.evaluator:main',
            'scripted_demo = franka_vla_sim.scripted_demo:main',
        ],
    },
)
