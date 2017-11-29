from setuptools import setup

setup(
    name='keras_gat',
    version='1.1',
    packages=['keras_gat'],
    install_requires=['keras'],  # Also requires tensorflow, but I don't want to mess up people's installs
    url='',
    license='',
    author='Daniele  Grattarola',
    author_email='daniele.grattarola@gmail.com',
    description='A Keras implementation of the Graph Attention Network model by'
                ' Velickovic et. al'
)
