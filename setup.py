"""
@author qx
@time 2024/06/18
"""

from setuptools import setup, find_packages

setup(
    name='qxanalyze',
    version='0.0.2',
    description="qx analyze",
    author='qx',
    author_email='lzj@qxtech.cc',
    python_requires='>=3.11',
    packages=find_packages(where="."),
    package_data={
        'qxanalyze': [
            'config/*.yaml',  
            'config/*.ini'
        ]
    },
    include_package_data=True,
    install_requires=[      
        "qxgentools",
    ],
    zip_safe=False,
)