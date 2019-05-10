# Learning Hawkes Processes Under Synchronization Noise

The code to run experiments with the DESYNC-MHP model is shipped here. Below are instructions to run the example code inside a [docker](https://www.docker.com/) container.

The `lib` folder contains all the internal code used in the paper (written in Python). Examples of code usage are provided in the `notebooks` folder as [jupyter](https://jupyter.org/) notebooks.


## 1. Installation

The code can be run over Docker. These instructions assume that [Docker Desktop](https://www.docker.com/products/docker-desktop) is installed on your computer and that a docker deamon is running.

1. We first need to build the docker image (this may take a few minutes to install and compile all dependencies).

    ```
    docker build -t desync-mhp .
    ```
    
    This creates the docker image `desync-mhp` with all the necessary dependencies.
    **Notice:** it may take a few minutes to build the image.
    

2. Now that the image is built, we can create a container to start a `jupyter` server on the 

    ```
    docker run -p 8888:8888 desync-mhp
    ```
    
This runs a container and exposes a `jupyter` server on port `8888`.


## 2. How to run the experiments?

The following instructions assume that the previous installation step was performed and that the `desync-mhp` container is running. A `jupyter` server can then be accessed by opening the following adress in a web browser:

    http://0.0.0.0:8888/?token=dummydummy
    

There are two notebooks illustrating the contributions on the paper.
    
### 2.1. DESYNC-MHP MLE on a toy example

The first notebook `1. DESYNC-MHP MLE on a toy example` takes the toy example used in the paper and applies both the classic maximum likelihood estimation and our DESYNC-MHP MLE approach to accurately recover the parameters of the of the model.

### 2.2. Effect of synchronization noise on the classic ML estimator

The second notebook `2. Effect of synchronization noise on the classic ML estimator` reproduces **Figure 1b** from the paper by varying the synchronization noise on a simple toy example to demonstrate the destructive effect of synchronization noise on the classic maximum likelihood estimation.
