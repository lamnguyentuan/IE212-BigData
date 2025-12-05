def remove_duplicate_and_overwrite(input_file):
    seen = set()
    unique_lines = []
    duplicate_lines = []

    # Đọc file
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.rstrip("\n")

            if clean_line in seen:
                duplicate_lines.append(clean_line)
            else:
                seen.add(clean_line)
                unique_lines.append(clean_line)

    # GHI ĐÈ LẠI FILE INPUT (chỉ giữ dòng unique)
    with open(input_file, "w", encoding="utf-8") as f:
        for line in unique_lines:
            f.write(line + "\n")

    # In báo cáo
    print("✅ Đã ghi đè file input.")
    if duplicate_lines:
        print("\n❌ Các dòng bị trùng đã bị xoá:")
        for line in duplicate_lines:
            print(line)
    else:
        print("\n✅ Không có dòng nào bị trùng.")


# =========================
# CHẠY CHƯƠNG TRÌNH
# =========================
if __name__ == "__main__":
    input_txt = "data-ingestion/tiktok_crawl/downloaded_videos.txt"   # đổi tên file tại đây
    remove_duplicate_and_overwrite(input_txt)
