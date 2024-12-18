from setuptools import setup

setup(
    name='mars_explorer',
    version='0.0.1',
    keywords='exploration, robotics, environment, agent, rl, openaigym, openai-gym, gym',
    description='Autonomous exploration of unknown planetary surface using lidar and RL techniques',
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.19.2',
        'pygame>=2.0.0'
    ],
    include_package_data=True,
    py_modules=[]
)
