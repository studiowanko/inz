# WUTSAT

This project lets you locate position of satellite by detecting area of Earth it has photographed!
It will be maintained in this [repository](https://github.com/studiowanko/inz) so check it out! If you get any problems 
with running this program contact me: 276446 (at) pw.edu.pl and title email ```WUTSAT help```

### Installation
To install this repository you need to run it under linux or [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) and follow this steps: 
- Install [Conda](https://pypi.org/project/conda/) on your machine.
- Go to the repository and create conda environment with ```conda env create -f environment.yml```
- Install [Jupitere Notebook](https://jupyter.org/install)

### Structure of project
In order to run the code properly in folder you have downloaded repo create 3 following directories:
```./img```, ```./hrit/decompressed-all```, ```./results```

### Getting files 
In order to properly run this code you need to:
* Place all photo from your satellite in ```./img``` directory
* All of the photos names and starting and ending time should be updated in file ```./inz/data/photos.csv```
* Also update ```./inz/data/data_sets.csv``` file with proper TLE and coresspoding ```set_id``` to ```id``` in ```./inz/data/photos.csv``` file
* At this stage data you need to download [TLE](https://www.space-track.org/documentation) data.
* Put TLE data in ```./inz/data/pw-sat2_tle.txt``` but remember this format is whitespace sensitive!
* Download raw photos from [EUMESAT](https://eoportal.eumetsat.int/cas/login). To get to know more how to get EUMESAT data read [this](https://satpy.readthedocs.io/en/latest/data_download.html). Ordering new data usually takes several hours. Be sure to decompress files with ```xRITDecompress``` [program](https://gitlab.eumetsat.int/open-source/PublicDecompWT). 
* Put decompressed data in ```./hrit/decompressed-all``` directory

### Running
Activate your environment, run Jupyter Notebook and open 2 notebooks in your browser. Be sure to read comments carefully!
```sh
$ conda activate inz_env
$ jupyter notebook
```
Run ```create_datasets``` and then ```calculate_corresponding_areas```  notebooks from ```notebooks``` directory.
In subdirectory you should have ```results``` folder with all results. In order to get specific data please read the source code. 
