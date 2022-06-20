# Action Detection in Smart Surveillance Systems - Web Application
![Logo](https://www.theigclub.com/wp-content/uploads/2020/12/IMG-20210123-WA0130.jpg)

# Table of Contents
1. [Introduction](#intro)
2.  [Features](#Features)
    * [Logs Page](#Logs)
    * [Visualizations Page](#Visualizations)
3.  [Demo](#demo)
4. [Screenshots](#screenshots)
5. [How to Run the Code](#run)
6. [Authors](#authors)
7. [🚀 About Us](#about)
8. [Contact](#contact)


## Introduction  <a name="intro"></a>
In this project, AI-based smart surveillance system capable of automatically detecting different actions was implemented on the Zewail City video surveillance system.
The system reached a recall of 80% and a precision of 70% without fine-tuning using custom data. The system provided different features for the end-user through its graphical interface allowing for easy and convenient data retrieval.


## Features  <a name="Features"></a>

**Logs page:** <a name="Logs"></a>


1- The user can view the history of actions along with some metadata:

    - The *id* of the *camera* that captured that action.
    - The *timestamp* indicating the start and end of the action.
    - The *Building* and *location* where the action happened.
    - The *confidence* of the predicted action.

2- The user can filter by:

    - Actions
    - Building
    - Location 
    - Camera 
    - Start, end date and time.

**Visualizations page:**  <a name="Visualizations"></a>

It provides different visualizations that deliver key insights to the user.

## Demo <a name="demo"></a>

Insert gif or link to demo


## Screenshots  <a name="screenshots"></a>

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## How to Run the Code <a name="run"></a>


1- Clone the project

```bash
  git clone https://github.com/mo-shaffei/action-detection-app
```

2- Go to the project directory

```bash
  cd action-detection-app
```

3- Install dependencies


- if you have CUDA

```bash
    pip install moviepy
    conda install -c pytorch torchvision cudatoolkit=10.2
    conda install -c conda-forge -c fvcore -c iopath fvcore=0.1.4 iopath
    pip install pytorchvideo
    conda install -c anaconda flask
    conda install -c anaconda pymongo
    conda install -c anaconda pandas
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    pip install scikit-video
    pip install opencv-python
```
    
- if you don't have CUDA

```bash
    pip install moviepy
    conda install -c pytorch torchvision cpuonly
    conda install -c conda-forge -c fvcore -c iopath fvcore=0.1.4 iopath
    pip install pytorchvideo
    conda install -c anaconda flask
    conda install -c anaconda pymongo
    conda install -c anaconda pandas
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    pip install scikit-video
    pip install opencv-python
```

4- Start the server

```bash
    flask run
```


## Authors <a name="authors"></a>

- [@Fatma Moanes](https://www.github.com/Fatma-Moanes) [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/fatma-moanes/)





## 🚀 About Us <a name="about"></a>
We are a team studying Communications and Information Engineering at Zewail City of Science and Technology. This project is our Graduation Project.

## Contact <a name="contact"></a>

To contact us:

fmoanesnoureldin@gmail.com


