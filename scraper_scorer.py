from curl_cffi import requests
import urllib.parse
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def fetch_note_data(keyword: str, max_pages: int = 2) -> pd.DataFrame:
    """noteの検索エンドポイントからデータを取得する（WAF完全回避＋セッション維持版）"""
    api_url = "https://note.com/api/v3/searches"
    search_url = f"https://note.com/search?context=note&q={urllib.parse.quote(keyword)}"

    # 変更点1: impersonate="chrome120" (より最新のバージョン) を指定
    session = requests.Session(impersonate="chrome120")

    # 変更点2: APIを叩く前にトップページへアクセスし、Bot判定をクリアしてCookieを取得する
    try:
        session.get("https://note.com/", timeout=10)
        time.sleep(random.uniform(1.0, 2.0)) # 人間らしいアクセス間隔を空ける
    except Exception:
        pass # トップページ取得に失敗しても処理は続行

    # 変更点3: API通信時にブラウザが自動付与するSec-Fetch系のヘッダーを完全再現
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": search_url,
        "Origin": "https://note.com",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    
    all_articles = []
    size = 20 # 1ページあたりの取得件数
    
    for page in range(max_pages):
        params = {
            "q": keyword,
            "context": "note",
            "size": size,
            "start": page * size
        }
        try:
            response = session.get(
                api_url, 
                headers=headers, 
                params=params, 
                timeout=15
            )
            
            if response.status_code == 403:
                st.error("⚠️ note側のセキュリティ(WAF)にブロックされました。環境（IPアドレス）が原因の可能性があります。")
                break
            elif response.status_code == 404:
                st.error("⚠️ noteのAPIが見つかりません。")
                break
            elif response.status_code != 200:
                st.error(f"⚠️ データ取得エラー: ステータスコード {response.status_code}")
                break
                
            data = response.json()
            response_data = data.get("data", {})
            
            if "notes" in response_data and isinstance(response_data["notes"], dict):
                notes = response_data["notes"].get("contents", [])
            elif "notes" in response_data and isinstance(response_data["notes"], list):
                notes = response_data["notes"]
            elif "contents" in response_data:
                notes = response_data["contents"]
            else:
                notes = []
            
            if not notes:
                break
                
            for item in notes:
                user_info = item.get("user", {})
                urlname = user_info.get("urlname", "unknown")
                key = item.get("key", "")
                
                article = {
                    "title": item.get("name"),
                    "like_count": item.get("likeCount", 0),
                    "author": urlname,
                    "created_at": item.get("createdAt", item.get("publishAt")),
                    "url": f"https://note.com/{urlname}/n/{key}" if key else ""
                }
                all_articles.append(article)
                
            time.sleep(random.uniform(2.0, 4.0)) # ページ間のウェイト
            
        except Exception as e:
            st.error(f"⚠️ 予期せぬエラーが発生しました: {e}")
            break
            
    return pd.DataFrame(all_articles)

def calculate_advanced_score(df: pd.DataFrame, weight_demand: float = 0.5, weight_density: float = 0.3, weight_recency: float = 0.2) -> pd.DataFrame:
    """統計とNLPを用いたブルーオーシャン・スコアリング"""
    if df.empty or len(df) < 2:
        return df

    log_likes = np.log1p(df['like_count'])
    min_like, max_like = log_likes.min(), log_likes.max()
    df['demand_score'] = (log_likes - min_like) / (max_like - min_like) if max_like > min_like else 0.5

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
    try:
        tfidf_matrix = vectorizer.fit_transform(df['title'].fillna(''))
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        avg_similarities = (cosine_sim.sum(axis=1) - 1) / (len(df) - 1)
        df['density_score'] = 1.0 - avg_similarities
    except Exception:
        df['density_score'] = 0.5

    now = datetime.now(timezone.utc)
    df['days_old'] = df['created_at'].apply(lambda x: (now - pd.to_datetime(x, utc=True)).days if pd.notnull(x) else 0)
    max_days = df['days_old'].max()
    df['recency_score'] = df['days_old'] / max_days if max_days > 0 else 0.5

    df['total_score'] = (
        (df['demand_score'] * weight_demand) +
        (df['density_score'] * weight_density) +
        (df['recency_score'] * weight_recency)
    ) * 100

    for col in ['total_score', 'demand_score', 'density_score', 'recency_score']:
        df[col] = df[col].round(1)
        if col != 'total_score':
             df[col] = (df[col] * 100).round(1)

    return df.sort_values(by='total_score', ascending=False).reset_index(drop=True)
