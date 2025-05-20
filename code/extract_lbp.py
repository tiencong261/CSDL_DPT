import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from pymongo import MongoClient

# Kết nối đến MongoDB
client = MongoClient('mongodb+srv://congdinh2412:SX3U5c8VLUTHZQfV@cluster0.3i7aqal.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['image_features']
collection = db['lbp_features']

def extract_lbp_features(image):
    """
    Trích xuất đặc trưng texture sử dụng LBP
    """
    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Tính LBP sử dụng scikit-image
    radius = 3
    n_points = 24
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Tính histogram của LBP
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist

def process_images():
    # Đường dẫn đến thư mục chứa ảnh
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(os.path.dirname(current_dir), 'Data')
    
    # Kiểm tra xem thư mục tồn tại không
    if not os.path.exists(image_dir):
        print(f"Thư mục ảnh không tồn tại: {image_dir}")
        return
    
    # Kiểm tra xem có dữ liệu trong MongoDB chưa
    existing_files = set()
    existing_count = collection.count_documents({})
    if existing_count > 0:
        for doc in collection.find({}, {"filename": 1}):
            existing_files.add(doc["filename"])
        print(f"Đã tìm thấy {len(existing_files)} ảnh đã được xử lý trước đó trong MongoDB")
    
    # Danh sách để lưu kết quả
    results = []
    
    # Lấy danh sách các file ảnh
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print("Đang trích xuất đặc trưng LBP...")
    print(f"Tìm thấy {len(image_files)} ảnh trong thư mục {image_dir}")
    
    # Xử lý từng ảnh
    for filename in tqdm(image_files):
        # Bỏ qua các file đã xử lý
        if filename in existing_files:
            continue
            
        try:
            # Đọc ảnh
            image_path = os.path.join(image_dir, filename)
            
            # Đọc ảnh bằng numpy trước
            image = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"Không thể đọc ảnh: {filename}")
                continue
                
            # Trích xuất đặc trưng LBP
            features = extract_lbp_features(image)
            
            # Tạo dictionary chứa tên file và các đặc trưng
            result = {'filename': filename}
            for i, feature in enumerate(features):
                result[f'feature_{i}'] = float(feature)
                
            results.append(result)
            
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {filename}: {str(e)}")
    
    if len(results) == 0:
        print("Không có ảnh mới nào cần xử lý")
        return
        
    # Lưu vào MongoDB
    collection.insert_many(results)
    print(f"\nĐã lưu {len(results)} bản ghi mới vào MongoDB collection 'lbp_features'")
    
    # In ra một vài ví dụ (chỉ in ra 5 đặc trưng đầu tiên)
    print("\nVí dụ về dữ liệu đã lưu (chỉ hiển thị 5 đặc trưng đầu tiên):")
    for doc in collection.find().limit(1):
        sample_doc = {k: doc[k] for k in list(doc.keys())[:6]}  # _id + filename + 4 features
        print(sample_doc)
    
    print(f"Tổng số bản ghi trong collection: {collection.count_documents({})}")

if __name__ == "__main__":
    process_images() 