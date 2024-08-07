from setuptools import setup, find_packages

setup(
    name="skin_condition_predictor",
    version="0.1",
    description="A Streamlit application to predict skin conditions using a fine-tuned MobileNetV2 model.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit",
        "tensorflow",
        "pandas",
        "numpy",
        "boto3",
        "Pillow",
        "scikit-learn"
    ],
    entry_points={
        'console_scripts': [
            'skin_condition_predictor = skin_condition_predictor.app:main'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

