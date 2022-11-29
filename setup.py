from setuptools import setup

setup(
    name='DecisionNet',
    version='0.1.0',
    description='Spiking Neural Network simulator for decision dynamics (using Brainpy)',
    url='https://github.com/throneofshadow/DecisionNet',
    author='Brett Nelson',
    author_email='bnelson@lbl.gov',
    license='BSD-3-Clause-LBNL',
    packages=['ReachSample'],
    install_requires=['brainpy'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)