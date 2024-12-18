# FashionMNIST Project for the course Machine Learning MAT3533 7
# Project purpose: Midterm
# Contributor
- Hoang Thiet Lam - Group Leader</br>
- Bui Huu Phuoc.<par></br>
- Pham Gia Nguyen.<par></br>
# Project Description
- Using FashionMNIST datasets, we implement 2 dimensionality reduction techniques: PCA and t-SNE. We compare their ability to visualize data </br>
- We also implement 2 classifier: Softmax Regression and Convolutional Neural Networks. We compare their ability to classify based on Accuracy, Precision, Recall metrics.
# How to read data
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
- Containing the source code for PCA visualization and analysis .ipynb file </br>
- Containing the plots from the code </br>
## PCA directory
- Containing the source code for t-SNE and CNN as .ipynb file </br>
- Containing the plots from the code </br>
## Logistic directory
- File _preprocessing_MNIST_Fashion_data_: Preprocess and explore the dataset.
- File _Test_with_784_dim_: Calculate metrics using various ratios and solvers on the original dataset..
- File _Test_with_46_dim_: Calculate metrics using various ratios and solvers on the reduced-dimension dataset.
- File _Find_c_each_solver_full_dim_: Find the appropriate parameter C for models showing signs of overfitting.
- File _test-l2-and-c_: Apply L2 regularization with the appropriate C parameter for models showing signs of overfitting.
- Workflow: _preprocessing_MNIST_Fashion_data_ -> _Test_with_784_dim, Test_with_46_dim_ -> _Find_c_each_solver_full_dim_ -> _test-l2-and-c_
- Containing the confusion matrix plot from the code </br>
