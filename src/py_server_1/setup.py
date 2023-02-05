from setuptools import setup

package_name = 'py_server_1'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kido',
    maintainer_email='deathfromthesky1998@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task2 = py_server_1.server_task2:main',
            'task3 = py_server_1.server_task3:main'
        ],
    },
)
