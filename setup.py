import setuptools

setuptools.setup(
    name="axon_id",
    version="0.0.1",
    author="Emily Joyce",
    author_email="emily.m.joyce1@gmail.com",
    description="tools used in identifying axons and dendrites of neurons",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    #project_urls={
        #"Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    #},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"axon_id": "models", "axon_id":"neuron_io", "axon_id":"groundtruth", "axon_id":"visualizations", "axon_id":"axon_worker.py"},
    packages=setuptools.find_packages(where = 'src'),
    python_requires=">=3.6",
)