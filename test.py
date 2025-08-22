import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm # Thêm thư viện để xem tiến trình

# --- CONFIGURATION ---
BATCH_SIZE = 32  # Bạn có thể điều chỉnh số này tùy vào RAM CPU của bạn
INPUT_FILE = "/home/toannk/Desktop/Code/ConvGQR-Rewrite_context/data/qrecc_train.json"
OUTPUT_FILE = "qrecc_train_translated_optimized.json"

# --- MODEL LOADING ---
print("Loading model and tokenizer...")
tokenizer_en2vi = AutoTokenizer.from_pretrained(
    "vinai/vinai-translate-en2vi-v2", src_lang="en_XX"
)
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")

# --- OPTIMIZATION 1: DYNAMIC QUANTIZATION FOR CPU ---
# Chuyển đổi trọng số của model sang int8 để tăng tốc tính toán trên CPU
# Lưu ý: Việc này có thể làm giảm một chút độ chính xác nhưng tăng tốc độ đáng kể.
print("Applying dynamic quantization for CPU acceleration...")
model_en2vi_quantized = torch.quantization.quantize_dynamic(
    model_en2vi, {torch.nn.Linear}, dtype=torch.qint8
)
print("Model quantized successfully.")

# Không cần chuyển model lên device cụ thể khi đã lượng tử hóa cho CPU
# device = torch.device("cpu")
# model_en2vi_quantized.to(device)


def translate_batch_en2vi(en_texts: list[str]) -> list[str]:
    """
    Hàm dịch một lô (batch) các câu từ tiếng Anh sang tiếng Việt.
    """
    # --- OPTIMIZATION 2: No_grad CONTEXT ---
    # Tắt việc tính toán gradient vì chúng ta đang trong quá trình suy luận (inference)
    with torch.no_grad():
        input_ids = tokenizer_en2vi(
            en_texts, padding=True, truncation=True, return_tensors="pt"
        )
        output_ids = model_en2vi_quantized.generate(
            **input_ids,
            decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True,
        )
    vi_texts = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    return vi_texts


# --- DATA PROCESSING ---
print(f"Loading data from {INPUT_FILE}...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

res = []
# Sử dụng tqdm để có thanh tiến trình trực quan
for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Translating batches"):
    batch_data = data[i : i + BATCH_SIZE]
    
    # Chuẩn bị một danh sách phẳng chứa tất cả các câu cần dịch trong lô
    texts_to_translate = []
    for d in batch_data:
        # Xử lý trường hợp Context có thể là list rỗng
        context_text = " ".join(d["Context"]) if d["Context"] else ""
        texts_to_translate.append(context_text)
        texts_to_translate.append(d["Question"])
        texts_to_translate.append(d["Rewrite"])

    # Dịch toàn bộ lô cùng một lúc
    translated_texts = translate_batch_en2vi(texts_to_translate)
    
    # Phân phối lại các câu đã dịch vào cấu trúc ban đầu
    for j, d in enumerate(batch_data):
        # Mỗi item trong batch_data tương ứng với 3 item trong translated_texts
        start_index = j * 3
        translated_context_str = translated_texts[start_index]
        
        res.append(
            {
                # Chuyển chuỗi context đã dịch về lại list (nếu cần) hoặc giữ nguyên
                "Context": [translated_context_str] if translated_context_str else [],
                "Question": translated_texts[start_index + 1],
                "Rewrite": translated_texts[start_index + 2],
            }
        )

# --- SAVING RESULTS ---
print(f"Saving translated data to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=4)

print("Translation complete!")