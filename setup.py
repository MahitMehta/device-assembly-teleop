from setuptools import find_packages, setup

package_name = "quest_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Mahit Mehta",
    description="Quest 3 controller teleop for the UR5e and Robotiq 2F-85",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "teleop_node = quest_teleop.nodes.teleop_node:main",
            "quest_sender = quest_teleop.tools.quest_sender:main",
        ],
    },
)
