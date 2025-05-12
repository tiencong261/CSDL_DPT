import os
import subprocess
import time

def run_script(script_name):
    """Chạy một script Python và hiển thị thông báo"""
    print(f"\n{'='*50}")
    print(f"Đang chạy {script_name}...")
    print(f"{'='*50}\n")
    
    # Chạy script
    subprocess.run(['python', script_name], check=True)
    
    print(f"\n{'='*50}")
    print(f"Hoàn thành {script_name}")
    print(f"{'='*50}\n")

def main():
    # Danh sách các script cần chạy
    scripts = [
        'extract_face.py',
        'extract_hog.py',
        'extract_hsv.py',
        'extract_lbp.py',
        'extract_gender.py',
        'combine_features_to_mongodb.py'
    ]
    
    start_time = time.time()
    
    # Chạy từng script
    for script in scripts:
        try:
            run_script(script)
        except subprocess.CalledProcessError as e:
            print(f"Lỗi khi chạy {script}: {str(e)}")
        except Exception as e:
            print(f"Lỗi không xác định khi chạy {script}: {str(e)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"Đã hoàn thành tất cả các script")
    print(f"Tổng thời gian: {total_time:.2f} giây ({total_time/60:.2f} phút)")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main() 