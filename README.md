
## Setup
List of commands to setup on Linux.

```
# install miniconda
curl [link to conda installer] miniconda.sh
chmod 755 miniconda.sh
./miniconda.ch

# install dependencies
conda install -c anaconda beautifulsoup4 lxml numpy
conda install -c pytorch pytorch torchvision torchaudio torchtext 

# make directories.
mkdir pages
mkdir models
```