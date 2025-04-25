from setuptools import setup, find_packages

setup(
    name="sensor_classification",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.7.0"
    ],
    entry_points={
        "console_scripts": [
            "sensor-cli=src.cli:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="다중 센서 데이터 분류 시스템",
    keywords="machine learning, lstm, sensor data, classification",
    python_requires=">=3.7",
)