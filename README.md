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

https://user-images.githubusercontent.com/57066226/175558538-00bfa5c3-3dae-4160-868b-5ebbd8ed68ed.mp4


## Screenshots  <a name="screenshots"></a>

![login2](https://user-images.githubusercontent.com/57066226/175560297-329e5343-85f8-492e-9743-5e621e687a2c.png)

![logs (1)](https://user-images.githubusercontent.com/57066226/175560338-8e0e81f3-92ae-4bc1-8362-e3dbdea0cb92.png)

![dashboard (1)](https://user-images.githubusercontent.com/57066226/175560346-7de17714-e984-4f29-9f3c-c2c0377978a8.PNG)

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
- [@Mohamed Elshaffei](https://www.github.com/mo-shaffei) [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohamed99elshaffei/)





## 🚀 About Us <a name="about"></a>
We are a group of students studying Communications and Information Engineering at Zewail City of Science and Technology in the final year. This project is our Graduation Project.

## Contact <a name="contact"></a>

To contact us:

fmoanesnoureldin@gmail.com

s-mohamed_chaffei@zewailcity.edu.eg

