#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# ============= تحميل النموذج والتوكنيزيشن ============
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("sentiment_model.h5")
    with open("tokenizer.json") as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

max_len = 100  # تأكد أنها نفس القيمة يلي استخدمتها وقت التدريب

# ============= تنظيف التعليق ============
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ============= أسباب محتملة ============
def extract_reason(comment):
    reasons = {
        "إعلانات": ["إعلانات", "اعلانات", "إعلان", "ads", "ad"],
        "بطء": ["بطيء", "بطيئ", "بطئ", "يعلق", "تأخير", "بطيئه"],
        "أخطاء": ["كراش", "خطأ", "تعليق", "ما يشتغل", "لا يعمل", "يهنق"],
        "سعر": ["غالي", "سعر", "فلوس", "دفع", "مبلغ"],
    }

    comment = comment.lower()
    for reason, keywords in reasons.items():
        for word in keywords:
            if word in comment:
                return reason
    return "غير محدد"

# ============= واجهة التطبيق ============
st.title("تحليل مشاعر تعليقات المستخدمين 🧠📊")
st.write("ارفع ملف CSV يحتوي على عمود اسمه `comment`")

uploaded_file = st.file_uploader("📤 ارفع الملف", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'comment' not in df.columns:
        st.error("❌ الملف يجب أن يحتوي على عمود باسم 'comment'")
    else:
        st.success("✅ تم تحميل الملف بنجاح")
        df['clean'] = df['comment'].apply(clean_text)
        sequences = tokenizer.texts_to_sequences(df['clean'])
        padded = pad_sequences(sequences, maxlen=max_len)

        predictions = model.predict(padded)
        df['prediction'] = (predictions > 0.5).astype(int)
        df['النتيجة'] = df['prediction'].apply(lambda x: "إيجابي" if x == 1 else "سلبي")
        df['السبب المحتمل'] = df.apply(lambda row: extract_reason(row['comment']) if row['النتيجة'] == "سلبي" else "", axis=1)

        st.subheader("📋 النتائج")
        st.dataframe(df[['comment', 'النتيجة', 'السبب المحتمل']])

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 تحميل النتائج كملف CSV", csv, "predicted_results.csv", "text/csv")

