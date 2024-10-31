import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread_collection
from skimage.transform import resize

# Giả sử bạn có 300 ảnh trong thư mục "dental_images"
# Chỉ định đường dẫn tới thư mục ảnh
image_folder = 'path_to_your_dental_images/*.jpg'

# Đọc ảnh
images = imread_collection(image_folder)
processed_images = []

# Tiền xử lý và thay đổi kích thước ảnh
for img in images:
    img_resized = resize(img, (64, 64))  # resize ảnh về kích thước 64x64
    processed_images.append(img_resized.flatten())  # chuyển thành vector

X = np.array(processed_images)
y = np.array([0] * 150 + [1] * 150)  # nhãn giả định: 0 và 1

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# CART (Gini)
cart_model = DecisionTreeClassifier(criterion="gini", random_state=42)
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)
print("Độ chính xác của CART (Gini Index):", metrics.accuracy_score(y_test, y_pred_cart))

# ID3 (Entropy)
id3_model = DecisionTreeClassifier(criterion="entropy", random_state=42)
id3_model.fit(X_train, y_train)
y_pred_id3 = id3_model.predict(X_test)
print("Độ chính xác của ID3 (Information Gain):", metrics.accuracy_score(y_test, y_pred_id3))
