from setuptools import setup

setup(
    name='linking_ds_vs_cp_proj',
    version='0.1.0',    
    description='linking_ds_vs_cp_proj',
    url='https://github.com/barbarabenato/linking_ds_vs_cp_proj.git',
    author='Bárbara C. Benato',
    author_email='barbara.benato@ic.unicamp.br',
    license='BSD 2-clause',
    packages=['linking_ds_vs_cp'],
    install_requires=[
                      'numpy',
                      'scipy',
                      'matplotlib',
                      'scikit-learn', 
                      'tensorflow',
                      'torch',
                      'umap'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)