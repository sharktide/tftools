from setuptools import setup, find_packages

setup(
    name="tftools",  # The name of your package
    version="0.1",    # The version of your package
    packages=find_packages(),  # Automatically find packages in the project
    install_requires=[  # List dependencies that your package needs
        "tensorflow",  
        "huggingface_hub",  
        # Add other dependencies here
    ],
    # Additional metadata about your package
    #author="Your Name",
    #author_email="your_email@example.com",
    description="Various Tensorflow Related Utilites. Not affiliated with Tensorflow or Tensorflow Addons or Huggingface in any way.",
    #long_description=open("README.md").read(),  # Add long description from README file
    #long_description_content_type="text/markdown",  # Content type for long description
    url="https://github.com/sharktide/tfaddons",  # URL to your package's repo
    classifiers=[
        "Programming Language :: Python :: 3.12",
        #"Operating System :: OS Independent",
    ]
)
