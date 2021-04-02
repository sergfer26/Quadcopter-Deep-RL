# Quadcopter-Deep-RL
Este proyecto consiste en entrenar a volar a un cuadricopetero por si mismo usando técnicas de deep reinforcement learning / This project consists of training a quadcopter to fly by itself using deep reinforcement learning techniques.    

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Python libraries:

- ``torch 1.6.0``
- ``gym 0.17.2``
- ``tqdm 4.45.0``
- ``reportlab 3.5.65``


```
pip install torch
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the trains and tests

Explain how to run the automated tests for this system

### DDPG + Supervised Learning

(initial phase) with simulation:

```
python3 trainSuper.py
```

### DDPG

Training with 18 states and simulation:

```
python3 trainDDPG.py
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* Python 3.7.7


## Authors

* **Sergio Miguel Fernández Martínez** 
* **Edgar Amilkar Gazque Espinosa de los Monteros**


## Acknowledgments

* We want to thank Antonio Capella Kort for his comments, his guidelines and  his time dedicated into this project.

* DDPG code: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b . 

## References

* Timothy P. Lillicrap. *Continuous control with deep reinforcement learning*. 9 Sep 2015.
