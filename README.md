# MAIS-202-F2023-ASL-Processing

<!-- ![](images/website.png) -->
Click the youtube link below for video of the demo
![](images/website.png)(https://www.youtube.com/watch?v=iEeXWWJMvew)

## About the Project

The challenge we are addressing revolves around the communication gap and lack of efficient tools for those who use American Sign Language (ASL). 

Our project aims to bridge this gap by providing a user-friendly platform for translating ASL letters into English letters with accuracy and speed. Leveraging a dataset of ASL gestures and corresponding English letters, our web app ensures a reliable and instant translation experience.

We recognize the communication barrier in various aspects of life, from education to professional interactions, and our web app provides a tool that promotes accessibility and inclusiveness.

# Launch the Web App

1. Install all the required packages by executing the following command in your terminal:

```
pip install -r requirements.txt
```

2. Change into the ASL-Processing-MLP directory of this repository with cd:

```
cd path/to/the/directory/ASL-Processing-MLP
```

3. Run the web app:

```
python web_app.py
```

4. Open your web browser and go to http://localhost:5000. Now you're all good to go!


# Load Dataset

1. Run [scr/loading_dataset.py](/src/loading_dataset.py) to load the data

2. With `load_data()` we load the [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet). 

3. With `image_to_landmarks()` we take the image in MediaPipe format, detects landmarks, and return 21 2d coordinates as an array

4. With `save_landmarks_disk.py` we save the 21 landmarks coordinations dataset

5. With `load_landmark_data.py` we split the dataset into training set (72%) validation set (18%) and testing set (10%)


<!-- # Choosen Hyperparameters Summary

| Hyperparameters         | Explanation                                              | Computed Result                                     |
|-------------------------|----------------------------------------------------------|-----------------------------------------------------|
| Learning Rate           | Pace the model learns the values of a parameter estimate | 0.001                                               |
| Weight Decay            | To penalizes large weights in the network                | 0.00001                                             |
| Dense Layers            | Number of linear layers                                  | 3                                                   |
| Neurons per Layers      | Number of neurons per dense layers                       | 1024                                                |
| Dropout                 | Number of neurons we leave out per dense layers          | 0.5                                                 |
 -->
