from setuptools import setup, find_packages

setup(
    name='pdfscan',       
    version='0.0.1',
    include_package_data=True,
    scripts=["src/pdfscan/client.sh"],
    entry_points={
        'console_scripts': [
            'pdfscan=pdfscan:main',
        ],
    },
    package_data={
        'pdfscan': ['xgboost-classifier.json'],  # Include JSON file in src package
    },
    install_requires=[
        'numpy',
        'pandas',
        'xgboost',
    ],
    author='Ganga Ram',
    author_email='ganga.ram@tii.ae',
    description='A Python package to generate secure configuration for systemd service.',
    long_description=open('README.md').read(),
    url='https://github.com/gangaram-tii/secure-systemd',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
