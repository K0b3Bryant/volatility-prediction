from setuptools import setup, find_packages

setup(
    name="monica_2000",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "ta",
        "tsfresh",
        "torch",
        "skorch",
        "optuna",
        "quantstats",
        "arch",
    ],
    entry_points={
        "console_scripts": [
            "monica2000=monica_2000.main:main",
        ]
    },
)
