# 🎬 Movie Recommender System

A content-based movie recommendation system built with Python and Streamlit.

## 🔍 Features
- Recommends 5 similar movies based on cast, director, genre, and keywords
- Clean and interactive UI with Streamlit
- Uses TF-IDF vectorization and cosine similarity

## 📁 Files
- `app.py`: Main Streamlit app
- `movies.csv`: Movie metadata
- `credits.zip`: Contains `credits.csv` (must be in the same folder)
- `requirements.txt`: All required Python packages

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run recco.py
