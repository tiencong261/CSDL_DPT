import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image
import shutil

def find_best_threshold(mask, min_ratio=0.15, max_ratio=0.7):
    h, w = mask.shape
    best_thresh = 0.5
    best_area = 0
    for thresh in np.arange(0.3, 0.71, 0.05):
        condition = mask > thresh
        area = np.sum(condition)
        ratio = area / (h * w)
        # Chọn mask có diện tích hợp lý nhất (ưu tiên gần 0.3-0.5)
        if min_ratio < ratio < max_ratio and area > best_area:
            best_thresh = thresh
            best_area = area
    return best_thresh

def process_image(image_path, output_path):
    try:
        # Sử dụng PIL để đọc ảnh
        pil_image = Image.open(image_path)
        # Chuyển đổi sang numpy array để OpenCV xử lý
        image = np.array(pil_image)
        # Nếu ảnh là RGBA (có kênh alpha), chuyển sang RGB
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        # Đảm bảo ảnh ở định dạng BGR cho OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Không thể đọc ảnh: {image_path}, lỗi: {str(e)}")
        return

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(image_rgb)
        mask = results.segmentation_mask

        # Tìm ngưỡng tốt nhất cho từng ảnh
        best_thresh = find_best_threshold(mask)
        condition = mask > best_thresh

        # Tạo ảnh nền trắng
        white_bg = np.ones(image.shape, dtype=np.uint8) * 255

        # Kết hợp ảnh gốc và nền trắng theo mask
        output_image = np.where(condition[..., None], image, white_bg)

        try:
            # Lưu ảnh bằng PIL để đảm bảo tương thích
            output_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            output_pil.save(output_path)
            print(f"Đã xử lý và lưu ảnh: {output_path}")
        except Exception as e:
            print(f"Lỗi khi lưu ảnh: {output_path}, lỗi: {str(e)}")

def process_directories(base_dir, input_dirs, output_dir):
    # Nếu thư mục đã tồn tại, xóa và tạo mới
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Tạo thư mục mới
    os.makedirs(output_dir)
    print(f"Đã tạo thư mục mới: {output_dir}")
    
    img_count = 1
    
    for input_dir in input_dirs:
        full_input_dir = os.path.join(base_dir, input_dir)
        print(f"Đang xử lý thư mục: {full_input_dir}")
        
        if not os.path.exists(full_input_dir):
            print(f"Thư mục không tồn tại: {full_input_dir}")
            continue
            
        files = os.listdir(full_input_dir)
        print(f"Số lượng file trong thư mục: {len(files)}")
        
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(full_input_dir, filename)
                # Thay đổi định dạng lưu từ jpg sang png
                output_filename = f"img{img_count}.png"
                output_path = os.path.join(output_dir, output_filename)
                process_image(input_path, output_path)
                img_count += 1

if __name__ == "__main__":
    # Đường dẫn tương đối từ vị trí của script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_directory = os.path.join(os.path.dirname(current_dir), "image")
    output_directory = os.path.join(os.path.dirname(current_dir), "Data_result")
    
    input_directories = ["Data1", "Data2"]
    
    process_directories(base_directory, input_directories, output_directory)
    print("Hoàn thành xử lý ảnh!")