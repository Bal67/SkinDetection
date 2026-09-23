from setuptools import find_packages, setup

setup(
    name="skin_detection",
    version="0.2.0",
    description="Research POC: skin-condition image classification evaluated across Fitzpatrick skin types.",
    packages=find_packages(include=["skin_detection", "skin_detection.*"]),
    install_requires=[
        "tensorflow>=2.16,<2.20",
        "numpy>=1.26",
        "pillow>=10.0",
        "h5py>=3.10",
    ],
    extras_require={
        "app": ["streamlit>=1.50"],
        "train": ["pandas>=2.1", "scikit-learn>=1.3", "matplotlib>=3.7", "requests>=2.31", "pytest>=8"],
        "s3": ["boto3"],
    },
    python_requires=">=3.10",
)
