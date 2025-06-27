import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import face_recognition
from pymongo import MongoClient

# Kết nối đến MongoDB
client = MongoClient('mongodb+srv://congdinh2412:SX3U5c8VLUTHZQfV@cluster0.3i7aqal.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['image_features']
collection = db['face_features']

def calculate_eye_ratio(left_eye, right_eye):
    """Tính tỷ lệ mắt"""
    left_width = np.linalg.norm(np.array(left_eye[0]) - np.array(left_eye[3]))
    right_width = np.linalg.norm(np.array(right_eye[0]) - np.array(right_eye[3]))
    return (left_width + right_width) / 2

def calculate_mouth_ratio(mouth):
    """Tính tỷ lệ miệng"""
    width = np.linalg.norm(np.array(mouth[0]) - np.array(mouth[6]))
    height = np.linalg.norm(np.array(mouth[3]) - np.array(mouth[9]))
    return width / height

def calculate_face_ratio(chin):
    """Tính tỷ lệ khuôn mặt"""
    width = np.linalg.norm(np.array(chin[0]) - np.array(chin[16]))
    height = np.linalg.norm(np.array(chin[8]) - np.array(chin[0]))
    return width / height

def extract_face_features(image):
    """
    Trích xuất đặc trưng dựa trên các điểm mốc trên khuôn mặt
    """
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) > 0:
        landmarks = face_recognition.face_landmarks(image, face_locations)[0]
        
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        eye_ratio = calculate_eye_ratio(left_eye, right_eye)
        
        mouth = landmarks['top_lip'] + landmarks['bottom_lip']
        mouth_ratio = calculate_mouth_ratio(mouth)
        
        chin = landmarks['chin']
        face_ratio = calculate_face_ratio(chin)
        
        features = {
            "eye_ratio": float(eye_ratio),
            "mouth_ratio": float(mouth_ratio),
            "face_ratio": float(face_ratio),
            "face": float((eye_ratio + mouth_ratio + face_ratio) / 3)
        }
        return features
    return None

def process_images():
    # Đường dẫn đến thư mục chứa ảnh
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(os.path.dirname(current_dir), 'Data')
    
    # Kiểm tra xem thư mục tồn tại không
    if not os.path.exists(image_dir):
        print(f"Thư mục ảnh không tồn tại: {image_dir}")
        return
    
    # Danh sách để lưu kết quả
    results = []
    
    # Lấy danh sách các file ảnh
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Xóa collection cũ nếu tồn tại
    collection.drop()
    
    print("Đang trích xuất đặc trưng khuôn mặt...")
    print(f"Tìm thấy {len(image_files)} ảnh trong thư mục {image_dir}")
    
    # Xử lý từng ảnh
    for filename in tqdm(image_files):
        try:
            # Đọc ảnh
            image_path = os.path.join(image_dir, filename)
            
            # Đọc ảnh bằng numpy trước
            image = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"Không thể đọc ảnh: {filename}")
                continue
                
            # Trích xuất đặc trưng khuôn mặt
            features = extract_face_features(image)
            
            if features is None:
                print(f"Không tìm thấy khuôn mặt trong ảnh: {filename}")
                continue
            
            # Tạo dictionary chứa tên file và các đặc trưng
            result = {
                'filename': filename,
                'eye_ratio': float(features["eye_ratio"]),
                'mouth_ratio': float(features["mouth_ratio"]),
                'face_ratio': float(features["face_ratio"]),
                'face': float(features["face"])
            }
                
            # Thêm vào danh sách kết quả
            results.append(result)
            
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {filename}: {str(e)}")
    
    # Lưu vào MongoDB
    if results:
        collection.insert_many(results)
        print(f"\nĐã lưu {len(results)} bản ghi vào MongoDB collection 'face_features'")
        
        # In ra một vài ví dụ
        print("\nVí dụ về dữ liệu đã lưu:")
        for doc in collection.find().limit(3):
            print(doc)
    else:
        print("Không có dữ liệu để lưu")

if __name__ == "__main__":
    process_images() 