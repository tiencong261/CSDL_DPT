import os
import io
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI
import matplotlib.pyplot as plt

# Import các hàm từ find_similar_images.py
from find_similar_images import (
    find_similar_images, 
    load_image,
    preprocess_image,
    extract_all_features,
    calculate_similarity,
    display_combined_features
)

app = Flask(__name__)
app.secret_key = "image_search_secret_key"
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Đảm bảo thư mục uploads tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Thư mục chứa dữ liệu ảnh
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')

# Các định dạng file ảnh được phép
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Kiểm tra xem có file trong request không
    if 'file' not in request.files and 'url' not in request.form:
        flash('Không có file hoặc URL được cung cấp', 'error')
        return redirect(request.url)
    
    # Xử lý trường hợp upload file
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Tìm ảnh tương tự
            try:
                similar_images = find_similar_images(filepath, num_results=3)
                if similar_images:
                    # Tạo kết quả hiển thị
                    result_image = create_result_display(filepath, similar_images)
                    return render_template('result.html', 
                                          result_image=result_image,
                                          similar_images=similar_images)
                else:
                    flash('Không tìm thấy ảnh tương tự', 'error')
                    return redirect(url_for('index'))
            except Exception as e:
                flash(f'Lỗi khi tìm ảnh tương tự: {str(e)}', 'error')
                return redirect(url_for('index'))
        else:
            flash('Định dạng file không được hỗ trợ', 'error')
            return redirect(url_for('index'))
    
    # Xử lý trường hợp URL
    elif 'url' in request.form and request.form['url'].strip() != '':
        url = request.form['url'].strip()
        
        # Tìm ảnh tương tự từ URL
        try:
            similar_images = find_similar_images(url, num_results=3)
            if similar_images:
                # Tạo kết quả hiển thị
                result_image = create_result_display(url, similar_images)
                return render_template('result.html', 
                                      result_image=result_image,
                                      similar_images=similar_images)
            else:
                flash('Không tìm thấy ảnh tương tự', 'error')
                return redirect(url_for('index'))
        except Exception as e:
            flash(f'Lỗi khi tìm ảnh tương tự từ URL: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    flash('Vui lòng chọn file hoặc nhập URL', 'error')
    return redirect(url_for('index'))

def create_result_display(input_image_path, similar_images):
    """
    Tạo ảnh kết quả hiển thị dưới dạng base64 để nhúng vào HTML
    """
    # Đường dẫn đến thư mục ảnh
    image_dir = DATA_DIR
    
    # Tạo layout 2 hàng để hiển thị tất cả ảnh
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Đọc và tiền xử lý ảnh đầu vào
    if input_image_path.startswith(('http://', 'https://')):
        # Xử lý URL giống như trong find_similar_images.py
        temp_input_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_input.jpg")
        import requests
        response = requests.get(input_image_path, stream=True)
        response.raise_for_status()
        
        with open(temp_input_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        local_input_path = temp_input_path
    else:
        local_input_path = input_image_path
    
    # Tiền xử lý ảnh đầu vào
    temp_output_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_output.png")
    try:
        from tien_xu_ly import process_image
        process_image(local_input_path, temp_output_path)
        
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
    except Exception as e:
        axs[0, 0].text(0.5, 0.5, f"Lỗi khi tiền xử lý ảnh: {str(e)}", 
                     horizontalalignment='center', verticalalignment='center')
        axs[0, 0].axis('off')
    
    # Xóa file tạm thời nếu là URL
    if input_image_path.startswith(('http://', 'https://')) and os.path.exists(local_input_path):
        os.remove(local_input_path)
    
    # Hiển thị các ảnh tương tự
    positions = [(0, 1), (1, 0), (1, 1)]  # Các vị trí còn lại trong layout 2x2
    
    for i, (filename, similarity) in enumerate(similar_images):
        if i >= 3:  # Chỉ hiển thị tối đa 3 ảnh tương tự
            break
            
        # Đường dẫn đến ảnh trong bộ dữ liệu
        image_path = os.path.join(image_dir, filename)
        
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
            row, col = positions[i]
            axs[row, col].text(0.5, 0.5, f"Không thể đọc ảnh {filename}", 
                             horizontalalignment='center', verticalalignment='center')
            axs[row, col].axis('off')
    
    plt.tight_layout()
    
    # Lưu ảnh vào buffer thay vì file
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    
    # Chuyển đổi buffer thành base64 để nhúng vào HTML
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)  # Đóng figure để giải phóng bộ nhớ
    
    return img_str

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 