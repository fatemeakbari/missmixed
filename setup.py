

from setuptools import setup, find_packages
print("Packages found:", find_packages())

setup(
    name="missmixed",
    version="0.1",
    packages=find_packages(),
    author="Mohammad Mahdi Kalhori",
    author_email="mohammad.mahdi.kalhor.99@gmail.com",
    description="A Novel Approach to Missing Value Imputation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=['data imputer'],
    # url="https://github.com/nydasco/package_publishing",

    python_requires='>=3.10',
    install_requires=[
            'openpyxl==3.1.3',
            'tqdm==4.67.1',
            'pandas>=2.0.0',
            'scikit-learn>=1.6.0',
        ],
    #openpyxl
)