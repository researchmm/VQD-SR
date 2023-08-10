from setuptools import setup
import os

def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires

if __name__ == '__main__':
    setup(
        name='vqdsr',
        version='1.0',
        description='Learning Data-Driven Vector-Quantized Degradation Model for Animation Video Super-Resolution',
        author='Zixi Tuo',
        author_email='zixit99@gmail.com',
        include_package_data=True,
        packages=['vqdsr'],
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='BSD-3-Clause License',
        setup_requires=['cython', 'numpy'],
        install_requires=get_requirements(),
        zip_safe=False)