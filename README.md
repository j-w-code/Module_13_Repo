# *Venture Funding with Deep Learning*
---

## Venture Funding with Deep Learning
This project covers the use of the neural networks and Deep Learning to make predictions about successful funding for startups. 

>"Learning never exhausts the mind."

## Technologies 

This project uses Pandas, Tensorflow, and SKlearn libraries to create dataframes, perform scaling operations on data, and initialize neural networking algorithms to make projections on data. 

[pandas](https://github.com/pandas-dev/pandas)
[SKLearn](https://github.com/scikit-learn/scikit-learn)
[Tensorflow](https://github.com/tensorflow/tensorflow)

### Installation Guide

In order to use this program please import and utilize the following libraries and dependencies: 

```python
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
```

## Usage 
the following blocks of code are fundamental in executing the program. 

```python 
enc = OneHotEncoder(sparse=False)
```
This creates an instance of the OneHotEncoder module for encoding categorical variables into numeric variables. 
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state)
```
This splits the dataset into training and testing data sites. The train_test_split function than processes the data and makes ready for calling by neural network. 
```python
number_input_features = len(X_train.loc[0])
```
This command sets the number of input features for the neural model to the legnth of the X_train data site. 

```python
nn = Sequential()
```
This command instantiates Sequential neural network model. 

```python
nn.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))
```
This command adds the first hidden layer of the neural network, it sets the input dimensions, and the activation function. Here set to Rectified Linear Unit ("relu"). 

```python
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
```
This command compiles the neural network. We are setting the loss function to "Binary CrossEntropy", the optimizer to "adam", and we're looking for "accuracy" with out metrics output. 

```python
fit_nn = nn.fit(X_train_scaled, y_train, epochs=)
```

This command fits the training data sites to the neural network and then runs projections on that data set to the number of epochs. 


![<alt text>](https://i.postimg.cc/xdBd5rBh/Screen-Shot-2022-07-17-at-2-32-44-PM.png)

This image shows the resulting accuracy and loss scores for the first of the nerual network projections run. 
    
    
## Contributors

Jeffrey J. Wiley Jr

## License

MIT


