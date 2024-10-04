import setuptools

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="powerlift",
    version="0.1.12",
    author="The InterpretML Contributors",
    author_email="interpret@microsoft.com",
    description="Interactive Benchmarking for Machine Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/interpretml/interpret",
    project_urls={
        "Bug Tracker": "https://github.com/interpretml/interpret/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.7",
    install_requires=[
        "SQLAlchemy >=1.4",
        "sqlalchemy-utils >=0.38",
        "tqdm",
        "fastparquet",
        "stopit",
        "pandas",
        "numpy",
        "pytz",
    ],
    extras_require={
        "datasets": [
            "pmlb >=1.0",
            "openml >=0.12",
            "catboost",
        ],
        "docker": [
            "docker",
        ],
        "postgres": [
            "psycopg2 >=2.9",
        ],
        "mssql": [
            "pyodbc",
        ],
        "aci": [
            "msrestazure",
            "azure-common",
            "azure-mgmt-sql",
            "azure-mgmt-resource",
            "azure-mgmt-containerinstance",
            "azure-mgmt-authorization",
            "azure-identity",
        ],
        "testing": [
            "pytest",
            "pytest-cov",
            "scikit-learn",
            "python-dotenv",
            "pytest-xdist",
        ],
    },
)
