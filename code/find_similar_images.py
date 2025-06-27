import cv2
import numpy as np
import os
#from pandas import pd  # Không sử dụng, xóa đi
#from sklearn.metrics.pairwise import cosine_similarity  # Không sử dụng, xóa đi
from pymongo import MongoClient
import face_recognition
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import time
import argparse
from PIL import Image
import io
import requests
from deepface import DeepFace
# Import từ file tien_xu_ly.py
from tien_xu_ly import process_image
from extract_face import calculate_eye_ratio, calculate_mouth_ratio, calculate_face_ratio, extract_face_features
from extract_hog import extract_hog_features
from extract_lbp import extract_lbp_features
from extract_hsv import extract_hsv_features
from extract_gender import extract_gender

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

def extract_all_features(image, predefined_gender=None):
    """
    Trích xuất tất cả các đặc trưng từ ảnh
    
    Parameters:
    - image: Ảnh cần trích xuất đặc trưng
    - predefined_gender: Giới tính đã biết trước (nếu có, ví dụ từ cơ sở dữ liệu)
    
    Returns:
    - Dictionary chứa các đặc trưng
    """
    features = {}
    
    # Trích xuất đặc trưng khuôn mặt
    face_features = extract_face_features(image)
    if face_features:
        features.update(face_features)
    
    # Trích xuất đặc trưng HOG
    hog_feature = extract_hog_features(image)
    features['hog'] = hog_feature
    
    # Trích xuất đặc trưng LBP
    lbp_feature = extract_lbp_features(image)
    features['lbp'] = lbp_feature
    
    # Trích xuất đặc trưng HSV
    hsv_feature = extract_hsv_features(image)
    features['hsv'] = hsv_feature
    
    # Xác định giới tính
    if predefined_gender is not None:
        # Sử dụng giới tính đã xác định trước đó
        gender = predefined_gender
        gender_text = "Nữ" if gender == 1 else "Nam"
        print(f"Sử dụng giới tính đã biết: {gender_text}")
    else:
        # Ưu tiên sử dụng DeepFace để xác định giới tính
        gender = extract_gender(image)
        
        # Nếu DeepFace không xác định được, sử dụng phương pháp dự đoán đơn giản
        if gender is None:
            gender = extract_gender(image)
            gender_text = "Nữ" if gender == 1 else "Nam"
            print(f"Giới tính nhận dạng từ ảnh đầu vào (dự đoán đơn giản): {gender_text}")
        else:
            gender_text = "Nữ" if gender == 1 else "Nam"
            # Chỉ hiển thị khi không dùng predefined_gender
            if predefined_gender is None:
                print(f"Giới tính nhận dạng từ ảnh đầu vào (DeepFace): {gender_text}")
    
    features['gender'] = gender
    
    return features

def calculate_similarity(input_features, db_features):
    """
    Tính độ tương đồng giữa đặc trưng của ảnh đầu vào và đặc trưng trong cơ sở dữ liệu
    """
    similarity = 0
    # Trọng số của từng loại đặc trưng trong việc tính độ tương đồng
    weight = {'face': 0.3, 'hog': 0.2, 'lbp': 0.2, 'hsv': 0.2, 'gender': 0.1}
    count = 0
    
    # Dictionary để lưu độ tương đồng của từng thuộc tính
    feature_similarities = {}
    
    # Tính điểm tương đồng cho từng loại đặc trưng
    for feature_type in ['face', 'hog', 'lbp', 'hsv']:
        if feature_type in input_features and feature_type in db_features:
            # Đảm bảo giá trị là float, nếu là array thì lấy giá trị trung bình
            input_val = input_features[feature_type]
            db_val = db_features[feature_type]
            try:
                if isinstance(input_val, np.ndarray):
                    input_val = float(np.mean(input_val))
                if isinstance(db_val, np.ndarray):
                    db_val = float(np.mean(db_val))
            except Exception:
                pass
            dist = abs(input_val - db_val)
            max_dist = 1.0  # Giá trị tối đa giả định
            norm_sim = 1 - min(dist / max_dist, 1.0)  # Chuẩn hóa về 0-1
            similarity += weight[feature_type] * norm_sim
            feature_similarities[feature_type] = norm_sim
            count += 1
    
    # Chuẩn hóa điểm tương đồng
    if count > 0:
        total_weight = sum([weight[key] for key in ['face', 'hog', 'lbp', 'hsv'] 
                          if key in input_features and key in db_features])
        final_similarity = similarity / total_weight
        
        # Trả về cả độ tương đồng tổng hợp và từng thuộc tính
        return final_similarity, feature_similarities
    return 0, {}

def load_image(image_path):
    """
    Đọc ảnh từ đường dẫn (file local hoặc URL)
    """
    try:
        if image_path.startswith(('http://', 'https://')):
            # Đọc ảnh từ URL
            print(f"Đang tải ảnh từ URL: {image_path}")
            response = requests.get(image_path)
            image = Image.open(io.BytesIO(response.content))
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 4:  # Nếu ảnh có kênh alpha
                image = image[:, :, :3]
            print(f"Đã tải ảnh thành công, kích thước: {image.shape}")
        else:
            # Đọc ảnh từ file local
            print(f"Đang đọc ảnh từ đường dẫn local: {image_path}")
            if os.path.isfile(image_path):
                image = np.fromfile(image_path, dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                print(f"Đã đọc ảnh thành công, kích thước: {image.shape}")
            else:
                print(f"Không tìm thấy file: {image_path}")
                print("Vui lòng kiểm tra lại đường dẫn hoặc thử với một ảnh khác.")
                return None
    except Exception as e:
        print(f"Lỗi khi đọc ảnh: {str(e)}")
        print("Vui lòng kiểm tra lại định dạng ảnh hoặc thử với một ảnh khác.")
        return None
    
    return image

def preprocess_image(image_or_path):
    """
    Tiền xử lý ảnh đầu vào sử dụng hàm process_image từ tien_xu_ly.py
    """
    # Tạo đường dẫn tạm thời cho ảnh đầu vào và ảnh đầu ra
    temp_input_path = "temp_input.png"
    temp_output_path = "temp_output.png"
    
    # Xử lý trường hợp đầu vào là URL
    if isinstance(image_or_path, str) and image_or_path.startswith(('http://', 'https://')):
        try:
            # Tải ảnh từ URL và lưu vào file tạm thời
            print(f"Tải ảnh từ URL và lưu vào file tạm thời: {temp_input_path}")
            response = requests.get(image_or_path, stream=True)
            response.raise_for_status()  # Kiểm tra lỗi HTTP
            
            with open(temp_input_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Gọi hàm process_image từ tien_xu_ly.py với file tạm thời
            process_image(temp_input_path, temp_output_path)
        except Exception as e:
            print(f"Lỗi khi tải ảnh từ URL: {str(e)}")
            # Nếu tải ảnh thất bại, trả về None
            return None
    # Xử lý trường hợp đầu vào là đường dẫn file local
    elif isinstance(image_or_path, str):
        try:
            # Gọi hàm process_image từ tien_xu_ly.py
            process_image(image_or_path, temp_output_path)
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh từ file local: {str(e)}")
            # Nếu xử lý thất bại, đọc ảnh gốc
            try:
                return cv2.imread(image_or_path)
            except:
                return None
    # Xử lý trường hợp đầu vào là mảng numpy (ảnh)
    else:
        try:
            # Lưu ảnh thành file tạm thời
            cv2.imwrite(temp_input_path, image_or_path)
            # Gọi hàm process_image từ tien_xu_ly.py
            process_image(temp_input_path, temp_output_path)
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh từ mảng numpy: {str(e)}")
            # Nếu xử lý thất bại, trả về ảnh gốc
            return image_or_path
    
    # Đọc ảnh đã xử lý
    if os.path.exists(temp_output_path):
        preprocessed_image = cv2.imread(temp_output_path)
        # Xóa file tạm thời
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        os.remove(temp_output_path)
        print("Đã hoàn thành tiền xử lý ảnh")
        return preprocessed_image
    else:
        print("Lỗi khi tiền xử lý ảnh")
        # Nếu không có ảnh đã xử lý, trả về None thay vì đường dẫn
        return None

def display_combined_features(features):
    """
    Hiển thị thông số combined của ảnh
    """
    print("\n" + "="*50)
    print("THÔNG SỐ ĐẶC TRƯNG CỦA ẢNH ĐẦU VÀO")
    print("="*50)
    
    # Hiển thị giới tính
    gender_text = "Nữ" if features.get('gender') == 1 else "Nam"
    print(f"Giới tính: {gender_text} ({features.get('gender')})")
    
    # Hiển thị đặc trưng khuôn mặt
    if 'face' in features:
        print("\nĐặc trưng khuôn mặt:")
        print(f"- Tỷ lệ mắt: {features.get('eye_ratio', 'N/A'):.4f}")
        print(f"- Tỷ lệ miệng: {features.get('mouth_ratio', 'N/A'):.4f}")
        print(f"- Tỷ lệ khuôn mặt: {features.get('face_ratio', 'N/A'):.4f}")
        print(f"- Giá trị face (trung bình): {features.get('face'):.4f}")
    
    # Hiển thị các đặc trưng khác
    print("\nĐặc trưng tổng hợp khác:")
    print(f"- HOG: {features.get('hog', 'N/A'):.4f}")
    print(f"- LBP: {features.get('lbp', 'N/A'):.4f}")
    print(f"- HSV: {features.get('hsv', 'N/A'):.4f}")
    
    print("="*50)

def find_similar_images(input_image_path, num_results=3):
    """
    Tìm các ảnh tương tự với ảnh đầu vào
    """
    try:
        # Lấy dữ liệu từ MongoDB
        print("Đang lấy dữ liệu đặc trưng ảnh từ MongoDB...")
        db_features = list(collection.find({}, {'_id': 0}))
        print(f"Đã lấy {len(db_features)} bản ghi từ cơ sở dữ liệu MongoDB")
        
        # Đếm số lượng ảnh theo giới tính
        male_count = sum(1 for feature in db_features if feature.get('gender', None) == 0)
        female_count = sum(1 for feature in db_features if feature.get('gender', None) == 1)
        print(f"Thống kê trong cơ sở dữ liệu: Tổng số ảnh: {len(db_features)}, Nam: {male_count}, Nữ: {female_count}")
        
        if not db_features:
            print("Không có dữ liệu đặc trưng ảnh trong cơ sở dữ liệu")
            print("Vui lòng chạy các script trích xuất đặc trưng và kết hợp đặc trưng trước.")
            return []
        
        # Đọc ảnh đầu vào
        input_image = load_image(input_image_path)
        if input_image is None:
            print("Không thể đọc ảnh đầu vào")
            return []
        
        # Tiền xử lý ảnh đầu vào
        print("\nĐang tiến hành tiền xử lý ảnh đầu vào...")
        preprocessed_image = preprocess_image(input_image_path)
        
        # Nếu tiền xử lý thất bại, sử dụng ảnh gốc
        if preprocessed_image is None:
            print("Tiền xử lý thất bại, sử dụng ảnh gốc")
            preprocessed_image = input_image
        
        # BƯỚC 1: XÁC ĐỊNH GIỚI TÍNH CỦA ẢNH ĐẦU VÀO
        print("Đang xác định giới tính của ảnh đầu vào...")
        
        # Kiểm tra xem ảnh có nằm trong cơ sở dữ liệu không
        input_filename = os.path.basename(input_image_path)
        db_image = next((img for img in db_features if img.get('filename') == input_filename), None)
        
        if db_image and 'gender' in db_image:
            # Nếu ảnh đã có trong cơ sở dữ liệu, lấy giới tính từ DB
            input_gender = db_image['gender']
            gender_text = "Nữ" if input_gender == 1 else "Nam"
            print(f"Giới tính lấy từ cơ sở dữ liệu: {gender_text}")
        else:
            # Nếu ảnh không có trong cơ sở dữ liệu, dùng DeepFace để xác định
            print("Ảnh không có trong cơ sở dữ liệu, sử dụng DeepFace để xác định giới tính...")
            input_gender = extract_gender(preprocessed_image)
            
            # Nếu DeepFace không xác định được, sử dụng phương pháp dự đoán đơn giản
            if input_gender is None:
                input_gender = extract_gender(preprocessed_image)
                gender_text = "Nữ" if input_gender == 1 else "Nam"
                print(f"Giới tính nhận dạng từ phương pháp dự đoán đơn giản: {gender_text}")
        
        # BƯỚC 2: LỌC ẢNH THEO GIỚI TÍNH
        print(f"Đang lọc ảnh theo giới tính...")
        filtered_db_features = [feature for feature in db_features if feature.get('gender', None) == input_gender]
        gender_text = "Nữ" if input_gender == 1 else "Nam"
        print(f"Đã lọc được {len(filtered_db_features)} ảnh có giới tính {gender_text} từ tổng số {len(db_features)} ảnh")
        
        if not filtered_db_features:
            print(f"Không tìm thấy ảnh nào có giới tính {gender_text} trong cơ sở dữ liệu.")
            print("Không thể tiếp tục tìm kiếm.")
            return []
        
        # BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG TỪ ẢNH ĐẦU VÀO
        print("Đang trích xuất các đặc trưng từ ảnh đầu vào (đã qua tiền xử lý)...")
        try:
            input_features = extract_all_features(preprocessed_image, input_gender)
            
            if not input_features:
                print("Không thể trích xuất đặc trưng từ ảnh đầu vào")
                print("Vui lòng thử lại với một ảnh khác có chứa khuôn mặt rõ ràng.")
                return []
        except Exception as e:
            print(f"Lỗi khi trích xuất đặc trưng: {str(e)}")
            print("Vui lòng thử lại với một ảnh khác.")
            return []
        
        # BƯỚC 4: TÍNH ĐỘ TƯƠNG ĐỒNG VỚI CÁC ẢNH CÙNG GIỚI TÍNH
        print("Đang tính toán độ tương đồng với các ảnh cùng giới tính...")
        similarities = []
        for db_feature in tqdm(filtered_db_features):
            # Loại bỏ chính ảnh đầu vào khỏi kết quả tìm kiếm
            if db_feature['filename'] == input_filename:
                continue
                
            # Tính độ tương đồng dựa trên các đặc trưng NGOẠI TRỪ giới tính
            # vì chúng ta đã lọc theo giới tính rồi
            similarity, feature_similarities = calculate_similarity(input_features, db_feature)
            similarities.append((db_feature['filename'], similarity, db_feature.get('gender', None), feature_similarities))
        
        # Sắp xếp theo độ tương đồng giảm dần
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # In thông tin chi tiết về các ảnh đã chọn
        print(f"\nĐã hoàn thành việc tính toán độ tương đồng cho {len(filtered_db_features)-1} ảnh")
        print(f"Top {num_results} ảnh tương đồng nhất (cùng giới tính {gender_text}):")
        print("-" * 80)
        for i, (filename, similarity, gender, feature_similarities) in enumerate(similarities[:num_results]):
            gender_text = "Nữ" if gender == 1 else "Nam"
            print(f"{i+1}. {filename}")
            print(f"   Giới tính: {gender_text}")
            print(f"   Độ tương đồng từng thuộc tính:")
            print(f"     - Face similarity: {feature_similarities.get('face', 0):.4f}")
            print(f"     - HOG similarity: {feature_similarities.get('hog', 0):.4f}")
            print(f"     - LBP similarity: {feature_similarities.get('lbp', 0):.4f}")
            print(f"     - HSV similarity: {feature_similarities.get('hsv', 0):.4f}")
            print("-" * 40)
        
        # Chỉ lấy tên file và độ tương đồng cho hàm trả về
        result = [(filename, similarity) for filename, similarity, _, _ in similarities[:num_results]]
        return result
    
    except Exception as e:
        print(f"Lỗi khi tìm ảnh tương tự: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def display_results(input_image_path, similar_images):
    """
    Hiển thị ảnh đầu vào (đã tiền xử lý) và 3 ảnh tương tự
    """
    # Đường dẫn đến thư mục ảnh
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(os.path.dirname(current_dir), 'Data')
    
    # Tạo cửa sổ hiển thị
    print("Đang hiển thị kết quả...")
    
    # Tạo layout 2 hàng để hiển thị tất cả ảnh
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Đọc và tiền xử lý ảnh đầu vào
    print("Đang đọc ảnh đầu vào...")
    
    # Xử lý trường hợp ảnh đầu vào là URL
    if isinstance(input_image_path, str) and input_image_path.startswith(('http://', 'https://')):
        try:
            # Tải ảnh từ URL
            temp_input_path = "temp_display_input.jpg"
            response = requests.get(input_image_path, stream=True)
            response.raise_for_status()
            
            with open(temp_input_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # Sử dụng đường dẫn tạm thời cho tiền xử lý
            local_input_path = temp_input_path
        except Exception as e:
            print(f"Lỗi khi tải ảnh từ URL để hiển thị: {str(e)}")
            # Nếu không thể tải ảnh, hiển thị thông báo lỗi
            for i in range(2):
                for j in range(2):
                    axs[i, j].text(0.5, 0.5, "Không thể tải ảnh", 
                                 horizontalalignment='center', verticalalignment='center')
                    axs[i, j].axis('off')
            plt.tight_layout()
            plt.savefig('ket_qua_tim_kiem.png', dpi=150)
            print("Đã lưu kết quả tìm kiếm vào file 'ket_qua_tim_kiem.png'")
            plt.show()
            return
    else:
        # Sử dụng đường dẫn gốc
        local_input_path = input_image_path
    
    # Tiền xử lý ảnh đầu vào
    temp_output_path = "temp_display_output.png"
    try:
        process_image(local_input_path, temp_output_path)
        
        # Đọc ảnh đã tiền xử lý
        if os.path.exists(temp_output_path):
            preprocessed_image = cv2.imread(temp_output_path)
            preprocessed_image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
            
            # Hiển thị ảnh đã tiền xử lý ở vị trí (0,0) - góc trên bên trái
            axs[0, 0].imshow(preprocessed_image_rgb)
            axs[0, 0].set_title("Ảnh đầu vào (đã tiền xử lý)", fontsize=12)
            axs[0, 0].axis('off')
            
            # Xóa file tạm
            os.remove(temp_output_path)
        else:
            # Nếu tiền xử lý thất bại, hiển thị ảnh gốc
            try:
                input_image = cv2.imread(local_input_path)
                if input_image is not None:
                    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                    axs[0, 0].imshow(input_image_rgb)
                    axs[0, 0].set_title("Ảnh đầu vào (gốc)", fontsize=12)
                    axs[0, 0].axis('off')
                else:
                    axs[0, 0].text(0.5, 0.5, "Không thể đọc ảnh", 
                                 horizontalalignment='center', verticalalignment='center')
                    axs[0, 0].axis('off')
            except Exception:
                axs[0, 0].text(0.5, 0.5, "Không thể đọc ảnh", 
                             horizontalalignment='center', verticalalignment='center')
                axs[0, 0].axis('off')
    except Exception as e:
        print(f"Lỗi khi tiền xử lý ảnh để hiển thị: {str(e)}")
        axs[0, 0].text(0.5, 0.5, "Lỗi khi tiền xử lý ảnh", 
                     horizontalalignment='center', verticalalignment='center')
        axs[0, 0].axis('off')
    
    # Xóa file tạm thời nếu là URL
    if isinstance(input_image_path, str) and input_image_path.startswith(('http://', 'https://')) and os.path.exists(temp_input_path):
        os.remove(temp_input_path)
    
    # Hiển thị các ảnh tương tự
    positions = [(0, 1), (1, 0), (1, 1)]  # Các vị trí còn lại trong layout 2x2
    
    for i, (filename, similarity) in enumerate(similar_images):
        if i >= 3:  # Chỉ hiển thị tối đa 3 ảnh tương tự
            break
            
        # Đường dẫn đến ảnh trong bộ dữ liệu
        image_path = os.path.join(image_dir, filename)
        print(f"Đang đọc ảnh tương tự {i+1}: {filename}")
        
        # Đọc ảnh
        try:
            # Dùng PIL/Pillow để đọc ảnh trực tiếp dưới dạng RGB
            image_pil = Image.open(image_path)
            image_rgb = np.array(image_pil)
        except Exception:
            try:
                # Fallback: Dùng OpenCV và chuyển đổi từ BGR sang RGB
                image = np.fromfile(image_path, dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception:
                # Nếu không thể đọc ảnh, hiển thị thông báo lỗi
                row, col = positions[i]
                axs[row, col].text(0.5, 0.5, f"Không thể đọc ảnh {filename}", 
                                 horizontalalignment='center', verticalalignment='center')
                axs[row, col].axis('off')
                continue
            
        if image_rgb is not None:
            # Hiển thị ảnh tương tự ở vị trí tương ứng
            row, col = positions[i]
            axs[row, col].imshow(image_rgb)
            axs[row, col].set_title(f"Ảnh {i+1}: {filename}\nĐộ tương đồng: {similarity:.4f}", fontsize=10)
            axs[row, col].axis('off')
        else:
            print(f"Không thể đọc ảnh {filename}")
            row, col = positions[i]
            axs[row, col].text(0.5, 0.5, f"Không thể đọc ảnh {filename}", 
                             horizontalalignment='center', verticalalignment='center')
            axs[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('ket_qua_tim_kiem.png', dpi=150)  # Lưu kết quả thành file ảnh với DPI cao hơn
    print("Đã lưu kết quả tìm kiếm vào file 'ket_qua_tim_kiem.png'")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Tìm kiếm ảnh tương tự dựa trên đặc trưng ảnh')
    parser.add_argument('--image', type=str, help='Đường dẫn đến ảnh đầu vào hoặc URL')
    parser.add_argument('--n', type=int, default=3, help='Số lượng ảnh tương tự cần tìm')
    
    args = parser.parse_args()
    
    # Nếu không có đường dẫn ảnh từ tham số, yêu cầu người dùng nhập
    image_path = args.image
    if not image_path:
        print("\nVui lòng nhập đường dẫn đến ảnh cần tìm:")
        print("- Có thể là đường dẫn local (ví dụ: ../Data/ten_anh.jpg)")
        print("- Hoặc URL ảnh trên internet (ví dụ: https://example.com/anh.jpg)")
        image_path = input("Đường dẫn ảnh: ").strip()
    
    num_results = args.n
    
    print("=" * 50)
    print(" CHƯƠNG TRÌNH TÌM KIẾM ẢNH TƯƠNG TỰ BẰNG MONGODB ")
    print("=" * 50)
    print(f"Ảnh đầu vào: {image_path}")
    print(f"Số lượng ảnh tương tự cần tìm: {num_results}")
    print("=" * 50)
    
    # Tìm các ảnh tương tự
    similar_images = find_similar_images(image_path, num_results)
    
    if similar_images:
        print("\nKẾT QUẢ TÌM KIẾM:")
        print("=" * 50)
        for i, (filename, similarity) in enumerate(similar_images):
            print(f"{i+1}. Tệp ảnh: {filename}")
            print(f"   Độ tương đồng: {similarity:.4f}")
            print("-" * 30)
        print("=" * 50)
        
        # Hiển thị kết quả
        display_results(image_path, similar_images)
    else:
        print("\nKhông tìm thấy ảnh tương tự nào.")
        print("Vui lòng thử lại với một ảnh khác hoặc kiểm tra lại cơ sở dữ liệu.")

if __name__ == "__main__":
    main()