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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_note_data(keyword: str, scraper_api_key: str, max_pages: int = 2) -> pd.DataFrame:
    proxy_auth = f"scraperapi.country_code=jp.premium=true:{scraper_api_key}"
    proxy_url = f"http://{proxy_auth}@proxy-server.scraperapi.com:8001"
    proxies = {"http": proxy_url, "https": proxy_url}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    all_articles = []
    size = 20
    api_url = "https://note.com/api/v3/searches"
    
    # 【追加】最大3回まで自動リトライを行う設定
    max_retries = 3 
    
    for page in range(max_pages):
        params = {"q": keyword, "context": "note", "size": size, "start": page * size}
        success = False
        
        # リトライ用のループ
        for attempt in range(max_retries):
            try:
                # タイムアウトを90秒に少し延長し、余裕を持たせる
                response = requests.get(api_url, headers=headers, params=params, proxies=proxies, verify=False, timeout=90)
                
                if response.status_code == 200:
                    success = True
                    break # データ取得成功！リトライループを抜ける
                    
                elif response.status_code == 403:
                    st.error("⚠️ 外部プロキシでのアクセスが拒否されました。APIキーの残高や有効性を確認してください。")
                    return pd.DataFrame(all_articles) # 致命的エラーなので即終了
                    
                elif response.status_code == 404:
                    st.error("⚠️ noteのAPIが見つかりません。")
                    return pd.DataFrame(all_articles) # 致命的エラーなので即終了
                    
                else:
                    # 499や500エラーの場合は、2秒待ってから別のIPでリトライする
                    time.sleep(2.0)
                    
            except requests.exceptions.RequestException:
                # ネットワークの切断やタイムアウト例外が出た場合もリトライ
                time.sleep(2.0)
                
        # 3回リトライしてもダメだった場合は、そのページだけ諦めて次に進む（ツール全体は止めない）
        if not success:
            st.warning(f"⚠️ 通信の混雑により一部のデータ（ページ{page+1}）がスキップされました。")
            continue
            
        try:
            data = response.json()
            response_data = data.get("data", {})
            
            notes = response_data.get("notes", {}).get("contents", []) if isinstance(response_data.get("notes"), dict) else response_data.get("notes", [])
            if not notes and "contents" in response_data:
                notes = response_data["contents"]
                
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
            st.error(f"⚠️ データ解析中に予期せぬエラーが発生しました: {e}")
            break
            
    return pd.DataFrame(all_articles)

def calculate_advanced_score(df: pd.DataFrame, weight_demand: float = 0.5, weight_density: float = 0.3, weight_recency: float = 0.2, user_context: str = "") -> pd.DataFrame:
    """統計とNLPを用いたブルーオーシャン・スコアリング（ノイズ排除フィルター搭載）"""
    if df.empty or len(df) < 2:
        return df

    log_likes = np.log1p(df['like_count'])
    min_like, max_like = log_likes.min(), log_likes.max()
    df['demand_score'] = (log_likes - min_like) / (max_like - min_like) if max_like > min_like else 0.5

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
    try:
        titles = df['title'].fillna('').tolist()
        
        if user_context:
            texts = titles + [user_context]
            tfidf_matrix_all = vectorizer.fit_transform(texts)
            tfidf_matrix = tfidf_matrix_all[:-1]
            context_vector = tfidf_matrix_all[-1]
            
            sim = cosine_similarity(tfidf_matrix, context_vector).flatten()
            df['relevance_score'] = sim
        else:
            tfidf_matrix = vectorizer.fit_transform(titles)
            df['relevance_score'] = 1.0

        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        avg_similarities = (cosine_sim.sum(axis=1) - 1) / (len(df) - 1) if len(df) > 1 else 0
        df['density_score'] = 1.0 - avg_similarities
        
    except Exception:
        df['density_score'] = 0.5
        df['relevance_score'] = 1.0

    now = datetime.now(timezone.utc)
    df['days_old'] = df['created_at'].apply(lambda x: (now - pd.to_datetime(x, utc=True)).days if pd.notnull(x) else 0)
    max_days = df['days_old'].max()
    df['recency_score'] = df['days_old'] / max_days if max_days > 0 else 0.5

    if 'relevance_score' in df.columns:
        max_rel = df['relevance_score'].max()
        if max_rel > 0:
            df['relevance_score'] = df['relevance_score'] / max_rel
        else:
            df['relevance_score'] = 1.0
    else:
        df['relevance_score'] = 1.0

    base_score = (
        (df['demand_score'] * weight_demand) +
        (df['density_score'] * weight_density) +
        (df['recency_score'] * weight_recency)
    )
    
    df['total_score'] = (base_score * (df['relevance_score'] ** 2)) * 100

    for col in ['total_score', 'demand_score', 'density_score', 'recency_score', 'relevance_score']:
        if col in df.columns:
            df[col] = df[col].round(1)
            if col != 'total_score':
                 df[col] = (df[col] * 100).round(1)

    return df.sort_values(by='total_score', ascending=False).reset_index(drop=True)
