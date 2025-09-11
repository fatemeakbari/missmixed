from setuptools import setup, find_packages

def get_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

packages = find_packages(include=["missmixed", "missmixed.*"])
print("Packages found:", packages)  # Debug print (helpful for you)

setup(
    name="missmixed",
    version="1.0.0",
    packages=packages,
    author="Mohammad Mahdi Kalhori",
    author_email="mohammad.mahdi.kalhor.99@gmail.com",
    description="A modular framework for missing value imputation using flexible iteration architectures.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    keywords=['missing data', 'missing data imputation', 'machine learning', 'data science', 'preprocessing'],
    url="https://github.com/MohammadKlhr/missmixed",

    python_requires='>=3.10',
    install_requires=[
        'openpyxl==3.1.3',
        'tqdm==4.67.1',
        'pandas>=2.0.0',
        "numpy>=1.22",
        'scikit-learn>=1.6.0',
    ],
    extras_require={
        "ml": ["xgboost>=2.0"],
        "deep": ["tensorflow>=2.0"]
    },
    entry_points={
        'console_scripts': [
            'missmixed = missmixed.run:main',
        ],
    },
)