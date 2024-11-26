from setuptools import setup, find_packages

setup(
    name='secure-service',       
    version='0.0.1',
    include_package_data=True,
    '''
    install_requires=[
        # Add other dependencies if needed
    ],
    '''
    scripts=["src/client.sh"],

    entry_points={
        'console_scripts': [
            'pdfclassifier=xgboost_classifier:main',
        ],
    },
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
