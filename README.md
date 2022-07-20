# AE4350 Deep reinforcement learning portfolio management system 

In the associated report a portfolio management system using deep reinforcement learning is designed as
practical assignment for the Bio-inspired Learning for Aerospace Applications (AE4350) course. This course is an an elective for Master's degree in Aerospace Engineering offered at the Delft University of Technology.
Active portfolio management is an important task for any serious investor, yet even beating a
buy-hold strategy can prove difficult. Therefore, a state-of-the-art intelligent system to tackle
this task seems worthwhile. The designed system is a soft actor-critic model applied to a discrete action space using the portfolio and time series state as input inspired by Saito et al [1] and presented in the image below. The system
is trained and tested on a 6-year period of S&P500 index time series data with its training
process being aided by a novel subsystem to improve training efficiency. In the report it is shown that the trained system is able achieve the intended goal and does beat the buy-hold strategy while invoking only few assumptions. All in all, it is concluded that in aproof-of-concept setting the fact that a system of such simple nature is able to achieve this feat
can be deemed a success.

This repository contains all of the files required to run the described system. The jupyter notebook has been adapted to work in a default Google Colab environment. Therefore, the reader is advised to clone this repository there and follow the steps as described in the notebook. 

For any specific questions about this system or a request for the report, please contact me at R.W.Vos@student.tudelft.nl or reinier.vos21@live.com

![actorCriticOverview](https://user-images.githubusercontent.com/99670985/180073536-b1f752d9-7370-4166-908b-ec4b5b4bb60a.jpg)

[1] Sean Saito, Yang Wenzhuo, and Rajalingappaa Shanmugamani. Python reinforcement learning projects: eight hands-on projects exploring reinforcement learning algorithms using TensorFlow. Packt Publishing Ltd, 2018.
