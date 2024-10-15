# Danh sách từ tiêu cực và tích cực
positive_words = [
    "thích", "tốt", "xuất sắc", "tuyệt vời", "tuyệt hảo", "đẹp",
    "ổn", "ngon", "hài lòng", "ưng ý", "hoàn hảo", "chất lượng",
    "thú vị", "nhanh", "tiện lợi", "dễ sử dụng", "hiệu quả",
    "ấn tượng", "nổi bật", "tận hưởng", "tốn ít thời gian",
    "thân thiện", "hấp dẫn", "gợi cảm", "tươi mới", "lạ mắt",
    "cao cấp", "độc đáo", "hợp khẩu vị", "rất tốt", "rất thích",
    "tận tâm", "đáng tin cậy", "đẳng cấp", "an tâm",
    "không thể cưỡng lại", "thỏa mãn", "thúc đẩy", "cảm động",
    "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội",
    "sáng tạo", "quý báu", "phù hợp", "hiếm có", "cải thiện",
    "hoà nhã", "chăm chỉ", "cẩn thận", "vui vẻ", "sáng sủa",
    "hào hứng", "đam mê", "vừa vặn", "đáng tiền", "nhiệt tình",
    "best", "good", "nghiện", "nhanh", "ngon nhất", "quá ngon",
    "quá tuyệt", "đúng vị", "điểm cộng", "thức ăn ngon",
    "khá ngon", "niềm nở", "rất thích", "đặc biệt", "không bị",
    "thơm", "ăn ngon", "cộng", "ủng_hộ quán", "ủng_hộ",
    "hấp_dẫn", "ấn_tượng", "thoải_mái", "quán ngon", "khen",
    "dài_dài", "tin_tưởng"
]

negative_words = [
    "kém", "tệ", "đau", "xấu", "dở", "ức", "buồn", "rối",
    "thô", "lâu", "chán", "tối", "ít", "mờ", "mỏng",
    "lỏng lẻo", "khó", "cùi", "yếu", "kém chất lượng",
    "không thích", "không thú vị", "không ổn", "không hợp",
    "không đáng tin cậy", "không chuyên nghiệp", "không phản hồi",
    "không an toàn", "không phù hợp", "không thân thiện",
    "không linh hoạt", "không đáng giá", "không ấn tượng",
    "không tốt", "chậm", "khó khăn", "phức tạp", "khó hiểu",
    "khó chịu", "gây khó dễ", "rườm rà", "khó truy cập",
    "thất bại", "tồi tệ", "khó xử", "không thể chấp nhận",
    "không rõ ràng", "không chắc chắn", "rối rắm", "không tiện lợi",
    "không đáng tiền", "chưa đẹp", "không đẹp", "bad",
    "thất vọng", "không ngon", "không hợp", "hôi", "trộm cướp",
    "không_ngon", "không_thích", "không_ổn", "không_hợp",
    "lần cuối", "cuối cùng", "quá tệ", "quá dở", "quá mắc",
    "cau có", "không đáng", "chả đáng", "điểm trừ",
    "thức ăn tệ", "đồ ăn tệ", "đợi lâu", "nhạt nhẽo",
    "không thoải mái", "không đặc sắc", "giá hơi", "chậm",
    "chậm chạm", "lâu", "quá lâu", "nhạt", "chờ",
    "ăn hơi", "khủng khiếp", "đợi", "nhạt", "thất_vọng",
    "bực_mình"
]

def find_negative_words(document):
    document_lower = document.lower()
    word_list = []
    for word in negative_words:
        if word in document_lower:
            print(f"Negative word found: {word}")
            word_list.append(word)
    return word_list

def find_positive_words(document):
    document_lower = document.lower()
    word_list = []
    for word in positive_words:
        if word in document_lower:
            print(f"Positive word found: {word}")
            word_list.append(word)
    return word_list

# Tài liệu mẫu
document = "Tôi rất thích món ăn này. Nó thật sự tuyệt vời nhưng cũng có một số điểm tệ."

# Gọi các hàm
negative_words_found = find_negative_words(document)
positive_words_found = find_positive_words(document)

# In kết quả
print("Negative words found:", negative_words_found)
print("Positive words found:", positive_words_found)
