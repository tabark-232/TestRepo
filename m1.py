#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os

# تحميل النموذج المدرب
model = tf.keras.models.load_model("model.h5")

# تحميل التوكن
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# تحميل الكلمات السلبية (اختياري)
with open("neg.txt", "r", encoding="utf-8") as f:
    negative_words = f.read().splitlines()

st.title("تحليل المشاعر ✨")
st.write("أدخل جملة وسنقوم بتحليل مشاعرها باستخدام نموذج مدرب! 😍😡")

text_input = st.text_input("اكتب جملة:")

if text_input:
    seq = tokenizer.texts_to_sequences([text_input])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)

    prediction = model.predict(padded)
    sentiment = "إيجابي 😄" if prediction > 0.5 else "سلبي 😢"

    st.subheader("النتيجة:")
    st.write(sentiment)

    neg_count = sum(word in text_input for word in negative_words)
    st.write(f"عدد الكلمات السلبية بالنص: {neg_count} 🧐")

