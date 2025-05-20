import pandas as pd
from pymongo import MongoClient
import time
import sys

# Kết nối đến MongoDB với timeout
try:
    print("Đang kết nối đến MongoDB...")
    client = MongoClient('mongodb+srv://congdinh2412:SX3U5c8VLUTHZQfV@cluster0.3i7aqal.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0', 
                        serverSelectionTimeoutMS=5000,  # 5 giây timeout
                        ssl=True,
                        tlsAllowInvalidCertificates=True)
    # Kiểm tra kết nối
    client.server_info()
    print("Kết nối MongoDB thành công!")
    db = client['image_features']
    collection = db['combined_features']
except Exception as e:
    print(f"Lỗi kết nối MongoDB: {str(e)}")
    print("Vui lòng kiểm tra kết nối mạng và thông tin đăng nhập MongoDB.")
    sys.exit(1)

def combine_features_from_mongodb():
    """
    Kết hợp các đặc trưng từ các collection MongoDB
    """
    try:
        # Lấy dữ liệu từ các collection
        print("Đang lấy dữ liệu từ collection hog_features...")
        hog_features = list(db['hog_features'].find({}, {'_id': 0}))
        print(f"Đã lấy {len(hog_features)} bản ghi từ hog_features")
        
        print("Đang lấy dữ liệu từ collection lbp_features...")
        lbp_features = list(db['lbp_features'].find({}, {'_id': 0}))
        print(f"Đã lấy {len(lbp_features)} bản ghi từ lbp_features")
        
        print("Đang lấy dữ liệu từ collection face_features...")
        face_features = list(db['face_features'].find({}, {'_id': 0}))
        print(f"Đã lấy {len(face_features)} bản ghi từ face_features")
        
        print("Đang lấy dữ liệu từ collection hsv_features...")
        hsv_features = list(db['hsv_features'].find({}, {'_id': 0}))
        print(f"Đã lấy {len(hsv_features)} bản ghi từ hsv_features")
        
        print("Đang lấy dữ liệu từ collection gender_features...")
        gender_features = list(db['gender_features'].find({}, {'_id': 0}))
        print(f"Đã lấy {len(gender_features)} bản ghi từ gender_features")
        
        # Kiểm tra xem có dữ liệu không
        if not hog_features or not lbp_features or not face_features or not hsv_features:
            print("Một hoặc nhiều collection không có dữ liệu")
            return []
        
        # Lấy danh sách tất cả các ảnh
        all_images = set()
        for features in [hog_features, lbp_features, face_features, hsv_features, gender_features]:
            for doc in features:
                if 'filename' in doc:
                    all_images.add(doc['filename'])
        
        print(f"Tổng số ảnh cần xử lý: {len(all_images)}")
        
        # Tạo dictionary để lưu trữ đặc trưng
        features_dict = {}
        
        # Khởi tạo dictionary với tên ảnh
        for image_name in all_images:
            features_dict[image_name] = {'filename': image_name}
        
        # Thêm đặc trưng HOG (mean và full vector)
        for doc in hog_features:
            if 'filename' not in doc:
                continue
            
            image_name = doc['filename']
            # Lấy tất cả các đặc trưng HOG
            feature_values = []
            for key, value in doc.items():
                if key.startswith('feature_') and isinstance(value, (int, float)):
                    feature_values.append(value)
            
            if feature_values:
                # Lưu giá trị trung bình
                features_dict[image_name]['hog'] = float(sum(feature_values) / len(feature_values))
        
        # Thêm đặc trưng LBP (mean)
        for doc in lbp_features:
            if 'filename' not in doc:
                continue
            
            image_name = doc['filename']
            # Tính trung bình của tất cả các đặc trưng LBP
            feature_values = []
            for key, value in doc.items():
                if key.startswith('feature_') and isinstance(value, (int, float)):
                    feature_values.append(value)
            
            if feature_values:
                features_dict[image_name]['lbp'] = float(sum(feature_values) / len(feature_values))
        
        # Thêm đặc trưng Face (mean của tất cả các đặc trưng khuôn mặt)
        for doc in face_features:
            if 'filename' not in doc:
                continue
            
            image_name = doc['filename']
            
            # Tính trung bình của tất cả các đặc trưng khuôn mặt
            face_values = []
            for key, value in doc.items():
                if key in ['eye_ratio', 'mouth_ratio', 'face_ratio'] and isinstance(value, (int, float)):
                    face_values.append(value)
            
            if face_values:
                features_dict[image_name]['face'] = float(sum(face_values) / len(face_values))
        
        # Thêm đặc trưng HSV (mean)
        for doc in hsv_features:
            if 'filename' not in doc:
                continue
            
            image_name = doc['filename']
            # Tính trung bình của tất cả các đặc trưng HSV
            feature_values = []
            for key, value in doc.items():
                if key.startswith('feature_') and isinstance(value, (int, float)):
                    feature_values.append(value)
            
            if feature_values:
                features_dict[image_name]['hsv'] = float(sum(feature_values) / len(feature_values))
        
        # Thêm đặc trưng Gender (Nam: 0, Nữ: 1)
        for doc in gender_features:
            if 'filename' not in doc or 'gender' not in doc:
                continue
            
            image_name = doc['filename']
            gender_value = 1 if doc['gender'] == 'Nữ' else 0  # Nam: 0, Nữ: 1
            features_dict[image_name]['gender'] = gender_value
        
        return list(features_dict.values())
    
    except Exception as e:
        print(f"Lỗi khi kết hợp đặc trưng: {str(e)}")
        return []

def main():
    try:
        # Kết hợp các đặc trưng từ MongoDB
        print("Bắt đầu kết hợp các đặc trưng...")
        features_list = combine_features_from_mongodb()
        
        if not features_list:
            print("Không có dữ liệu để lưu")
            return
            
        # Xóa collection cũ nếu tồn tại
        print("Xóa collection cũ nếu tồn tại...")
        collection.drop()
        
        # Lưu vào MongoDB
        print(f"Lưu {len(features_list)} bản ghi vào MongoDB...")
        collection.insert_many(features_list)
        print(f"Đã lưu {len(features_list)} bản ghi vào MongoDB collection 'combined_features'")
        
        # In ra một vài ví dụ
        print("\nVí dụ về dữ liệu đã lưu:")
        for doc in collection.find().limit(3):
            print(doc)
    
    except Exception as e:
        print(f"Lỗi trong quá trình thực thi: {str(e)}")

if __name__ == "__main__":
    main() 