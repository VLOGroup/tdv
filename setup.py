from setuptools import setup


setup(name='tdv',
      install_requires=[
          'numpy', 'scikit-image', 'imageio', 'matplotlib',
          'optox  @ https://github.com/VLOGroup/optox/archive/master.zip'],
      packages=['tdv'])
