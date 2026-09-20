import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CLEAN_FILE = BASE_DIR / 'data/processed/players_stat_cleaned.csv'
REPORT_FILE = BASE_DIR / 'data/raw/missing_values_report.csv'


def create_missing_report(file_paths):
    reports = []
    for path in file_paths:
        if not path.exists():
            print(f'Không tìm thấy file: {path}')
            continue
        missing_info = pd.read_csv(path, low_memory=False).isnull().sum()
        missing_info = missing_info[missing_info > 0]
        if missing_info.empty:
            continue
        missing_df = missing_info.rename('missing_count').reset_index(names='column_name')
        missing_df['table_name'] = path.name
        missing_df['missing_percentage'] = round(missing_df['missing_count'] / len(pd.read_csv(path)) * 100, 2)
        reports.append(missing_df)
    return pd.concat(reports, ignore_index=True)[['table_name', 'column_name', 'missing_count', 'missing_percentage']] if reports else pd.DataFrame()


def save_missing_report(file_paths, output_file=REPORT_FILE):
    report = create_missing_report(file_paths)
    if report.empty:
        print('Không có bảng nào bị khuyết dữ liệu.')
        return report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_file, index=False)
    print(f'Đã xuất báo cáo tại: {output_file}')
    print(report.sort_values('missing_percentage', ascending=False).head(10).to_string(index=False))
    return report

def analyze_missing_clusters(file_path):
    print(f"--- PHÂN TÍCH CỤM DỮ LIỆU KHUYẾT THIẾU ---")
    
    # Đọc dữ liệu
    df = pd.read_csv(file_path, low_memory=False)
    total_rows = len(df)
    print(f"Tổng số dòng ban đầu của bảng: {total_rows:,}\n")

    # 1. Định nghĩa các nhóm dựa trên file báo cáo của bạn
    # Nhóm 23%: Thường là các trận đấu cũ trước 2023 web chưa cập nhật đủ
    cluster_cols = ['kast_t', 'rating2_t', 'acs_t', 'adr_t', 'hsp_t'] 
    
    # Nhóm 1%: Lỗi vặt, tuyển thủ dự bị không thi đấu
    individual_cols = ['kills_all', 'fb_all', 'fd_all', 'team'] 

    # 2. KIỂM TRA CỤM (CROSS-CHECK) CHO NHÓM 23%
    print("1. KIỂM TRA NHÓM LỖI NẶNG (~23%):")
    missing_cluster_mask = df[cluster_cols].isnull()
    
    # Đếm số dòng mà TẤT CẢ các cột trong cụm đều rỗng cùng lúc
    all_null = missing_cluster_mask.all(axis=1).sum()
    # Đếm số dòng mà CÓ ÍT NHẤT 1 cột trong cụm bị rỗng
    any_null = missing_cluster_mask.any(axis=1).sum()
    
    print(f"- Số dòng bị khuyết TẤT CẢ {len(cluster_cols)} chỉ số cùng lúc: {all_null:,}")
    print(f"- Số dòng bị khuyết ÍT NHẤT 1 chỉ số: {any_null:,}")
    
    if all_null == any_null:
        print("=> ✅ KẾT LUẬN: Tuyệt vời! Các chỉ số này ĐI CHUNG MỘT CỤM 100%. Xóa 1 cột cũng bằng xóa 5 cột.\n")
    else:
        print("=> ⚠️ KẾT LUẬN: Có sự sai lệch nhẹ, chúng không đi theo cụm hoàn toàn.\n")

    # 3. MÔ PHỎNG CẮT BỎ DỮ LIỆU (SIMULATE DROP)
    print("2. MÔ PHỎNG TỔN THẤT DỮ LIỆU NẾU DÙNG LỆNH DROPNA:")
    
    # Kịch bản A: Chỉ xóa nhóm lỗi vặt 1%
    df_drop_ind = df.dropna(subset=individual_cols)
    loss_ind = total_rows - len(df_drop_ind)
    print(f"- Kịch bản A (Chỉ xóa lỗi vặt lẻ tẻ): Mất {loss_ind:,} dòng ({round(loss_ind/total_rows*100, 2)}%).")

    # Kịch bản B: Xóa TẤT CẢ (Cả cụm 23% và lỗi vặt 1%)
    df_drop_all = df.dropna(subset=cluster_cols + individual_cols)
    loss_all = total_rows - len(df_drop_all)
    print(f"- Kịch bản B (Xóa sạch mọi lỗi): Mất {loss_all:,} dòng ({round(loss_all/total_rows*100, 2)}%).")
    print(f"  -> Số dòng sạch 100% còn lại để đưa vào Machine Learning: {len(df_drop_all):,}\n")


def main():
    save_missing_report([CLEAN_FILE])
    analyze_missing_clusters(CLEAN_FILE)

if __name__ == "__main__":
    main()