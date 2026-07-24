from setuptools import setup, find_packages

setup(
    name="dermatology-fairness-auditor",
    version="0.2",
    description=(
        "A Streamlit tool that audits dermatology AI model predictions for "
        "accuracy disparities across Fitzpatrick skin-tone groups. Not a "
        "diagnostic tool; evaluates predictions the user already generated "
        "elsewhere."
    ),
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "scipy",
        "requests",
        # Retained for the legacy/scripts model-training and demo-generation
        # code (see README "Background"); not needed by the audit engine itself.
        "tensorflow",
        "boto3",
        "Pillow",
        "scikit-learn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

