# coding=utf-8
# author=UlionTse

import re
import pathlib
import setuptools


NAME = 'mlgb'
PACKAGE = 'mlgb'
AUTHOR = 'UlionTse'
AUTHOR_EMAIL = 'uliontse@outlook.com'
HOMEPAGE_URL = 'https://github.com/uliontse/mlgb'
DESCRIPTION = 'MLGB is a library that includes many models of CTR Prediction & Recommender System by TensorFlow & PyTorch.'
LONG_DESCRIPTION = pathlib.Path('README.md').read_text(encoding='utf-8')
VERSION = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', pathlib.Path('mlgb/__init__.py').read_text(), re.M).group(1)


setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license='Apache-2.0',
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_dir={'mlgb': 'mlgb'},
    url=HOMEPAGE_URL,
    project_urls={
        'Source': 'https://github.com/UlionTse/mlgb',
        'Changelog': 'https://github.com/UlionTse/mlgb/blob/main/change_log.txt',
        'Documentation': 'https://github.com/UlionTse/mlgb/blob/main/README.md',
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        # 'Programming Language :: Python :: 3.8',  # 2024.10
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords=['Machine Learning', 'Deep Learning', 'CTR Prediction', 'Recommender System'],
    install_requires=[
        'numpy>=1.23.0',
        'pandas>=1.5.0',
        'scikit-learn>=1.3.0',
        'torch>=2.1.0',
        'tensorflow>=2.10.0',
        # 'tensorflow-addons>=0.21.0',
    ],
    python_requires='>=3.9',
    extras_require={'pypi': ['build>=0.10.0', 'twine>=4.0.2']},
    zip_safe=False,
)














