import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tempfile

st.set_page_config(page_title="OMR Test Değerlendirme", layout="centered")
st.title("📝 Optik Form Değerlendirme Sistemi")

st.markdown("""
Bu uygulama, çoktan seçmeli sınavlar için taranmış optik formları analiz eder ve cevapları cevap anahtarıyla karşılaştırarak puanlama yapar.
""")

# Cevap anahtarı girişi
answer_key_text = st.text_input("📌 Cevap Anahtarı (örn: ABCDABCDABCD...)", max_chars=30)

# Görsel yükleme
uploaded_file = st.file_uploader("📷 Taranmış Optik Formu Yükleyin (PNG veya JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file and answer_key_text:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    image = cv2.imread(tmp_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubble_contours = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.2 and 15 <= w <= 50 and 15 <= h <= 50:
            bubble_contours.append(contour)

    def sort_contours(cnts, method="top-to-bottom"):
        reverse = False
        i = 1
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        elif method == "left-to-right" or method == "right-to-left":
            i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
        return cnts

    bubble_contours_sorted = sort_contours(bubble_contours, method="top-to-bottom")

    question_bubbles = []
    for i in range(0, len(bubble_contours_sorted), 4):
        cnts_group = sort_contours(bubble_contours_sorted[i:i+4], method="left-to-right")
        question_bubbles.append(cnts_group)

    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    student_answers = []
    for q_group in question_bubbles:
        max_nonzero = 0
        selected_option = None
        for idx, cnt in enumerate(q_group):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            if total > max_nonzero:
                max_nonzero = total
                selected_option = idx
        letter = ['A', 'B', 'C', 'D'][selected_option] if selected_option is not None else "-"
        student_answers.append(letter)

    answer_key = list(answer_key_text.upper())
    results = []
    correct_count = 0
    for i, (student, correct) in enumerate(zip(student_answers, answer_key)):
        is_correct = student == correct
        if is_correct:
            correct_count += 1
        results.append({
            'Soru No': i+1,
            'Öğrenci Cevabı': student,
            'Doğru Cevap': correct,
            'Doğru mu?': '✔' if is_correct else '✘'
        })

    df = pd.DataFrame(results)
    df.loc[len(df.index)] = ['Toplam', '', '', f'{correct_count} / {len(answer_key)}']

    st.success("✅ Değerlendirme Tamamlandı")
    st.dataframe(df)
    st.download_button("📥 Sonuçları İndir (Excel)", data=df.to_csv(index=False), file_name="sinav_sonuclari.csv", mime="text/csv")
