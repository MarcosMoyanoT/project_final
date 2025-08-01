from setuptools import find_packages, setup

setup(
    name='fraud_detection',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'matplotlib',
        'optuna',
        'imblearn',
    ],
)
