import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse


def filter_missing_images(json_file: str, images_folder: str, output_file: str = None) -> Dict[str, Any]:
    """
    Lọc các nhãn không có ảnh từ file JSON.
    
    Args:
        json_file: Đường dẫn tới file JSON (vd: train.json)
        images_folder: Đường dẫn tới thư mục chứa các ảnh
        output_file: Đường dẫn tới file JSON kết quả (tùy chọn)
    
    Returns:
        Dict chứa thông tin về việc lọc:
        - 'total_images': Tổng số nhãn trong JSON ban đầu
        - 'missing_images': Số nhãn không có ảnh
        - 'valid_images': Số nhãn có ảnh
        - 'missing_files': Danh sách tên file bị thiếu
        - 'filtered_data': Dữ liệu JSON đã lọc
    """
    
    # Kiểm tra file JSON tồn tại
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"File JSON không tồn tại: {json_file}")
    
    # Kiểm tra thư mục ảnh tồn tại
    if not os.path.isdir(images_folder):
        raise NotADirectoryError(f"Thư mục ảnh không tồn tại: {images_folder}")
    
    # Đọc file JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Lấy danh sách các ảnh trong thư mục
    image_files = set(os.listdir(images_folder))
    
    # Lọc các nhãn có ảnh
    original_images = data.get('images', [])
    filtered_images = []
    missing_files = []
    
    for image_entry in original_images:
        file_name = image_entry.get('file_name')
        
        if file_name in image_files:
            filtered_images.append(image_entry)
        else:
            missing_files.append(file_name)
    
    # Tạo dữ liệu JSON đã lọc
    filtered_data = data.copy()
    filtered_data['images'] = filtered_images
    
    # Lưu kết quả nếu output_file được cung cấp
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        print(f"✓ File JSON đã lọc được lưu tại: {output_file}")
    
    # Tạo thông tin báo cáo
    result = {
        'total_images': len(original_images),
        'valid_images': len(filtered_images),
        'missing_images': len(missing_files),
        'missing_files': missing_files,
        'filtered_data': filtered_data
    }
    
    return result


def print_report(result: Dict[str, Any]) -> None:
    """In báo cáo kết quả lọc"""
    print("\n" + "="*60)
    print("BÁOCAO LỌC ẢNH")
    print("="*60)
    print(f"Tổng số nhãn trong JSON: {result['total_images']}")
    print(f"Số nhãn có ảnh: {result['valid_images']}")
    print(f"Số nhãn không có ảnh: {result['missing_images']}")
    
    if result['missing_images'] > 0:
        print(f"\n⚠ Các file ảnh bị thiếu ({result['missing_images']} file):")
        for i, file_name in enumerate(result['missing_files'][:10], 1):
            print(f"  {i}. {file_name}")
        
        if result['missing_images'] > 10:
            print(f"  ... và {result['missing_images'] - 10} file khác")
    else:
        print("\n✓ Tất cả các ảnh đều tồn tại!")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Lọc các nhãn không có ảnh từ file JSON'
    )
    parser.add_argument(
        'json_file',
        help='Đường dẫn tới file JSON (vd: train.json)'
    )
    parser.add_argument(
        'images_folder',
        help='Đường dẫn tới thư mục chứa các ảnh (vd: images/)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Đường dẫn file JSON đầu ra (tùy chọn)',
        default=None
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Hiển thị báo cáo chi tiết'
    )
    
    args = parser.parse_args()
    
    try:
        # Thực hiện lọc
        result = filter_missing_images(
            args.json_file,
            args.images_folder,
            args.output
        )
        
        # In báo cáo
        if args.verbose or args.output:
            print_report(result)
        else:
            print_report(result)
        
    except FileNotFoundError as e:
        print(f"❌ Lỗi: {e}")
        return 1
    except NotADirectoryError as e:
        print(f"❌ Lỗi: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi: File JSON không hợp lệ - {e}")
        return 1
    except Exception as e:
        print(f"❌ Lỗi không mong muốn: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
