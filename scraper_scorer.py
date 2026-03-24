import requests
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def fetch_note_data(keyword: str, max_pages: int = 2) -> pd.DataFrame:
    """noteの検索エンドポイントからデータを取得する"""
    api_url = "https://note.com/api/v2/search/notes"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    
    all_articles = []
    
    for page in range(1, max_pages + 1):
        params = {"q": keyword, "sort": "popular", "page": page}
        try:
            response = requests.get(api_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            notes = data.get("data", {}).get("notes", [])
            
            if not notes:
                break
                
            for item in notes:
                article = {
                    "title": item.get("name"),
                    "like_count": item.get("likeCount", 0),
                    "author": item.get("user", {}).get("urlname"),
                    "created_at": item.get("createdAt"),
                    "url": f"https://note.com/{item.get('user', {}).get('urlname')}/n/{item.get('key')}"
                }
                all_articles.append(article)
                
            time.sleep(random.uniform(1.5, 3.0)) # IPバン対策のウェイト
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                st.error("⚠️ アクセスが集中しているため、note側で一時的な制限がかかりました。数分後に再度お試しください。")
            else:
                st.error(f"⚠️ データ取得エラー: {e}")
            break
        except Exception as e:
            st.error(f"⚠️ 予期せぬエラーが発生しました: {e}")
            break
            
    return pd.DataFrame(all_articles)

def calculate_advanced_score(df: pd.DataFrame, weight_demand: float = 0.5, weight_density: float = 0.3, weight_recency: float = 0.2) -> pd.DataFrame:
    """統計とNLPを用いたブルーオーシャン・スコアリング"""
    if df.empty or len(df) < 2:
        return df

    # 1. 需要スコア (対数変換)
    log_likes = np.log1p(df['like_count'])
    min_like, max_like = log_likes.min(), log_likes.max()
    df['demand_score'] = (log_likes - min_like) / (max_like - min_like) if max_like > min_like else 0.5

    # 2. 競合密度スコア (TF-IDF文字N-gram)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
    try:
        tfidf_matrix = vectorizer.fit_transform(df['title'].fillna(''))
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        avg_similarities = (cosine_sim.sum(axis=1) - 1) / (len(df) - 1)
        df['density_score'] = 1.0 - avg_similarities
    except Exception:
        df['density_score'] = 0.5

    # 3. リプレイス機会スコア (経過日数)
    now = datetime.now(timezone.utc)
    df['days_old'] = df['created_at'].apply(lambda x: (now - pd.to_datetime(x, utc=True)).days if pd.notnull(x) else 0)
    max_days = df['days_old'].max()
    df['recency_score'] = df['days_old'] / max_days if max_days > 0 else 0.5

    # 4. 総合スコアの算出
    df['total_score'] = (
        (df['demand_score'] * weight_demand) +
        (df['density_score'] * weight_density) +
        (df['recency_score'] * weight_recency)
    ) * 100

    # 表示用に丸める
    for col in ['total_score', 'demand_score', 'density_score', 'recency_score']:
        df[col] = df[col].round(1)
        if col != 'total_score':
             df[col] = (df[col] * 100).round(1)

    return df.sort_values(by='total_score', ascending=False).reset_index(drop=True)