# Action Detection in Smart Surveillance Systems - Web Application
![Logo](https://www.theigclub.com/wp-content/uploads/2020/12/IMG-20210123-WA0130.jpg)

# Table of Contents
1. [Introduction](#intro)
2. [System Block Diagram](#block)
3.  [Features](#Features)
    * [Logs Page](#Logs)
    * [Visualizations Page](#Visualizations)
4.  [Demo](#demo)
5. [Screenshots](#screenshots)
6. [How to Run the Code](#run)
7. [Authors](#authors)
8. [🚀 About Us](#about)
9. [Contact](#contact)


## Introduction  <a name="intro"></a>
In this project, AI-based smart surveillance system capable of automatically detecting different actions was implemented on the Zewail City video surveillance system.
The system reached a recall of 80% and a precision of 70% without fine-tuning using custom data. The system provided different features for the end-user through its graphical interface allowing for easy and convenient data retrieval.


## System Block Diagram    <a name="block"></a>
![block diagram](https://user-images.githubusercontent.com/57066226/175679868-76c93abe-95d8-42b1-afb2-c2f155e13759.png)


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

![photo_2022-06-19_12-21-10](https://user-images.githubusercontent.com/57066226/175560675-a78188f7-52fc-4a21-b13e-e986c980b2d5.jpg)

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
- [@Mahmoud Ashraf](https://github.com/MahmoudAshraf97) [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mahmoudashraf1997/)

- [@Touka Mohamed](https://github.com/Touka-Mohamed) [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/touka-mohamed-1354071b0/)



## 🚀 About Us <a name="about"></a>
We are a group of students studying Communications and Information Engineering at Zewail City of Science and Technology in the final year. This project is our Graduation Project.

## Contact <a name="contact"></a>

To contact us:

fmoanesnoureldin@gmail.com

s-mohamed_chaffei@zewailcity.edu.eg

hassouna97.ma@gmail.com

s-touka99-mohamed@zewailcity.edu.eg 

