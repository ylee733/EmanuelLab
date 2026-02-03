import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='analyze2p',
                 version='0.1',
                 description='Scripts for analyzing multielectrode array recordings',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 author='Ronghao Zhang',
                 author_email='ronghao.zhang@emory.edu',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 install_requires=['numpy','matplotlib','scipy','re','glob','os',
                 'math','sys'])