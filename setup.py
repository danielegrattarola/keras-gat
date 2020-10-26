from setuptools import setup

setup(
    name='keras_gat',
    version='1.4',
    packages=['keras_gat'],
    install_requires=['keras<2.3', 'tensorflow<2'],
    url='https://github.com/danielegrattarola/keras-gat',
    license='MIT',
    author='Daniele Grattarola',
    author_email='daniele.grattarola@gmail.com',
    description='A Keras implementation of the Graph Attention Network by Velickovic et. al'
)
