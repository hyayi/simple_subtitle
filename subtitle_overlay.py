import whisper
from transformers import MarianMTModel, MarianTokenizer
import torch
from PyQt5 import QtWidgets, QtCore
import sys
import threading
import os

# =====================
# 모델 로드
# =====================
def load_models():
    whisper_cache_dir = os.path.expanduser("~/.cache/whisper")
    translator_cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")

    print("📥 Whisper base 모델 로드 중...")
    if not os.path.exists(whisper_cache_dir):
        print("🔽 Whisper 모델이 없어서 다운로드 시작...")
    whisper_model = whisper.load_model("base")

    print("📥 MarianMT 번역기 로드 중...")
    if not os.path.exists(translator_cache_dir):
        print("🔽 MarianMT 번역 모델이 없어서 다운로드 시작...")
    model_name = "Helsinki-NLP/opus-mt-ko-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translator_model = MarianMTModel.from_pretrained(model_name)

    print("✅ 모델 로드 완료!")
    return whisper_model, tokenizer, translator_model

# =====================
# PyQt5 자막 오버레이
# =====================
class SubtitleOverlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Subtitle Overlay')
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setGeometry(100, 600, 1600, 200)

        self.ko_label = QtWidgets.QLabel(self)
        self.ko_label.setStyleSheet("color: white; font-size: 32px; background-color: rgba(0, 0, 0, 128);")
        self.ko_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ko_label.setGeometry(0, 0, 1600, 80)

        self.en_label = QtWidgets.QLabel(self)
        self.en_label.setStyleSheet("color: yellow; font-size: 28px; background-color: rgba(0, 0, 0, 128);")
        self.en_label.setAlignment(QtCore.Qt.AlignCenter)
        self.en_label.setGeometry(0, 80, 1600, 60)

    def update_subtitles(self, ko_text, en_text):
        self.ko_label.setText(ko_text)
        self.en_label.setText(en_text)

# =====================
# Whisper 음성 인식 및 번역
# =====================
def recognize_and_translate(overlay, whisper_model, tokenizer, translator_model):
    print("🎤 Whisper 음성 인식 시작 (Ctrl+C로 종료)...")
    while True:
        try:
            result = whisper_model.transcribe("default", language="ko", fp16=False)
            text_ko = result['text'].strip()
            print(f"[한글] {text_ko}")

            inputs = tokenizer([text_ko], return_tensors="pt", padding=True)
            translated = translator_model.generate(**inputs)
            text_en = tokenizer.decode(translated[0], skip_special_tokens=True)
            print(f"[영어] {text_en}")

            overlay.update_subtitles(text_ko, text_en)
        except KeyboardInterrupt:
            print("⏹️ 종료")
            break

# =====================
# PyQt5 앱 실행
# =====================
def run_app():
    whisper_model, tokenizer, translator_model = load_models()

    app = QtWidgets.QApplication(sys.argv)
    overlay = SubtitleOverlay()
    overlay.show()

    threading.Thread(target=recognize_and_translate, args=(overlay, whisper_model, tokenizer, translator_model), daemon=True).start()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_app()
