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
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dự án FashionMNIST cho môn học Machine Learning MAT3533 7  
## Mục đích dự án: Kiểm tra giữa kỳ  

### Thành viên đóng góp  
- **Hoàng Thiết Lâm** - Trưởng nhóm  
- **Bùi Hữu Phước**  
- **Phạm Gia Nguyên**  

## Mô tả dự án  
- Sử dụng bộ dữ liệu **FashionMNIST**, chúng tôi triển khai 2 kỹ thuật giảm chiều dữ liệu: **PCA** và **t-SNE**. Chúng tôi so sánh khả năng trực quan hóa dữ liệu của hai phương pháp này.  
- Ngoài ra, chúng tôi cũng triển khai 2 bộ phân loại: **Softmax Regression** và **Mạng Nơ-ron Tích Chập (CNN)**. Chúng tôi so sánh khả năng phân loại dựa trên các chỉ số **Độ chính xác (Accuracy)**, **Độ chính xác (Precision)**, và **Độ nhạy (Recall)**.  

## Hướng dẫn đọc dữ liệu  
Để tải và chia bộ dữ liệu **FashionMNIST**, bạn có thể sử dụng đoạn mã sau:  

```python
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Tải bộ dữ liệu FashionMNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
data = fashion_mnist.load_data()
(X_train, y_train), (X_test, y_test) = data

# Chia thành tập huấn luyện và tập kiểm định: 48,000 huấn luyện, 12,000 kiểm định, 10,000 kiểm tra
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
```
# Thư mục
## Thư mục T-SNE CNN
- Chứa mã nguồn phân tích và trực quan hóa PCA (.ipynb file).
- Chứa các biểu đồ được sinh ra từ mã nguồn.
## Thư mục PCA
- Chứa mã nguồn liên quan đến t-SNE và CNN (.ipynb file).
- Chứa các biểu đồ được sinh ra từ mã nguồn.
## Thư mục Logistic
- File _preprocessing_MNIST_Fashion_data_: Tiền xử lý và khám phá dữ liệu.
- File _Test_with_784_dim_: Tính toán các chỉ số với các tỷ lệ và solver từ tập dữ liệu gốc.
- File _Test_with_46_dim_: Tính toán các chỉ số với các tỷ lệ và solver từ tập dữ liệu đã giảm chiều.
- File _Find_c_each_solver_full_dim_: Tìm tham số C phù hợp cho các mô hình có dấu hiệu overfit.
- File _test-l2-and-c_: Áp dụng hiệu chỉnh L2 với tham số C phù hợp cho các mô hình có dấu hiệu overfit.
- Cách thức chạy: _preprocessing_MNIST_Fashion_data_ → _Test_with_784_dim_, _Test_with_46_dim_ → _Find_c_each_solver_full_dim_ → _test-l2-and-c_
- Chứa các biểu đồ ma trận nhầm lẫn (Confusion Matrix) được sinh ra từ mã nguồn.

