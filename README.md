# FashionMNIST Project for the course Machine Learning MAT3533 7
# Project purpose: Midterm
# Contributor
Hoang Thiet Lam - Group Leader</br>
Bui Huu Phuoc.<par></br>
Pham Gia Nguyen.<par></br>
# Project Description
Using FashionMNIST datasets, we implement 2 dimensionality reduction techniques: PCA and t-SNE. We compare their ability to visualize data </br>
We also implement 2 classifier: Softmax Regression and Convolutional Neural Networks. We compare their ability to classify based on Accuracy, Precision, Recall metrics.
# How to read data
## How to Read Data
To load and split the FashionMNIST dataset, you can use the following code: </br>
```python
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load FashionMNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
data = fashion_mnist.load_data()
(X_train, y_train), (X_test, y_test) = data

# Split into training and validation sets: 48000 training, 12000 Validation, 10000 Test
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
```
# Directories
## T-SNE CNN directory
Containing the source code for PCA visualization and analysis .ipynb file </br>
Containing the plots from the code </br>
## PCA directory
Containing the source code for t-SNE and CNN as .ipynb file </br>
Containing the plots from the code </br>
## Logistic directory
Containing the source code for t-SNE and CNN as .ipynb file </br>
Each of the files are the fine-tuning hyperparameters process for data splits / reduced dimension etc... </br>
Containing the confusion matrix plot from the code </br>
