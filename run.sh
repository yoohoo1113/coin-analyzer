#!/bin/bash

# Streamlit 암호화폐 분석기 실행 스크립트

echo "🪙 Streamlit 암호화폐 분석기 시작 중..."

# 필요한 패키지 설치
echo "📦 패키지 설치 중..."
pip install -r requirements.txt

# Streamlit 앱 실행
echo "🚀 앱 실행 중..."
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
