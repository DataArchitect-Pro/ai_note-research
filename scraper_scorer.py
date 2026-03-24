import requests
import urllib.parse
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import urllib3

# セキュリティ警告（ScraperAPI経由時のSSL警告）を非表示にする
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_note_data(keyword: str, scraper_api_key: str, max_pages: int = 2) -> pd.DataFrame:
    """ScraperAPIをプロキシモードで使用し、URLエンコードエラーを完全に回避する"""
    
    # 変更点1: REST方式を廃止し、プロキシURLを構築
    # country_code=jp を追加し、必ず日本のIPを経由させる（海外IP弾きを回避）
    proxy_auth = f"scraperapi.country_code=jp.premium=true:{scraper_api_key}"
    proxy_url = f"http://{proxy_auth}@proxy-server.scraperapi.com:8001"
    
    proxies = {
        "http": proxy_url,
        "https": proxy_url
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    
    all_articles = []
    size = 20
    api_url = "https://note.com/api/v3/searches"
    
    for page in range(max_pages):
        params = {
            "q": keyword,
            "context": "note",
            "size": size,
            "start": page * size
        }
        
        try:
            # 変更点2: proxies引数を使用。プロキシによるSSL傍受を許可するため verify=False を指定
            response = requests.get(
                api_url,
                headers=headers,
                params=params,
                proxies=proxies,
                verify=False,
                timeout=60
            )
            
            if response.status_code == 403:
                st.error("⚠️ 外部プロキシでのアクセスが拒否されました。APIキーの残高や有効性を確認してください。")
                break
            elif response.status_code == 404:
                st.error("⚠️ noteのAPIが見つかりません。")
                break
            elif response.status_code != 200:
                st.error(f"⚠️ データ取得エラー: ステータスコード {response.status_code}\n詳細: {response.text[:100]}")
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
                
            time.sleep(random.uniform(1.0, 2.0))
            
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
