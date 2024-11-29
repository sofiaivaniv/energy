from setuptools import setup, find_packages

setup(
    name="energy_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'scikit-learn>=0.24.0',
        'statsmodels>=0.12.0',
    ],
    authors="Софія Івнів, Вікторія Яковлєва",
    authors_email="sofiia.ivaniv.sa.2022@lpnu.ua, viktoriia.yakovlieva.sa.2022@lpnu.ua",
    description="A package for energy consumption analysis and forecasting",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sofiaivaniv/energy.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires='>=3.8',
)