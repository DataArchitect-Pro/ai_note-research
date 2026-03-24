import streamlit as st
import pandas as pd
import time
from scraper_scorer import fetch_note_data, calculate_advanced_score
from content_generator import generate_content_plan

st.set_page_config(page_title="noteブルーオーシャン発掘ツール", layout="wide")

# --- 1. 簡易パスワード認証 ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.markdown("### 🔒 ツールへのログイン")
        pwd = st.text_input("購入者限定パスワードを入力してください", type="password")
        if pwd == "note-ai-2026": # ※運用時に任意の文字列に変更してください
            st.session_state["password_correct"] = True
            st.rerun()
        elif pwd:
            st.error("パスワードが間違っています。")
        st.stop()

check_password()

# --- 2. サイドバー（入力UI） ---
st.sidebar.title("⚙️ 設定・入力")

# 【追加部分】2つのAPIキーを入力させる
st.sidebar.markdown("**1. APIキーの設定**")
api_key = st.sidebar.text_input("OpenAI APIキー (sk-...)", type="password", help="構成案の自動生成に使用します")
scraper_api_key = st.sidebar.text_input("ScraperAPIキー", type="password", help="noteのデータ取得（ブロック回避）に使用します")

st.sidebar.markdown("---")
st.sidebar.markdown("**2. リサーチ条件**")
keyword = st.sidebar.text_input("🔍 リサーチキーワード", placeholder="例：経理 エクセル 時短")
target_reader = st.sidebar.text_area("👤 ターゲット読者像", placeholder="例：手作業で毎月残業している30代経理")
user_strength = st.sidebar.text_area("💪 あなたの本業・強み", placeholder="例：メーカー経理歴10年、VBAとPythonが書ける")

st.sidebar.markdown("---")
with st.sidebar.expander("詳細スコアリング設定"):
    max_pages = st.slider("データ取得量（ページ数）", 1, 5, 2)
    weight_demand = st.slider("需要（スキ数）の重み", 0.0, 1.0, 0.5)
    weight_density = st.slider("競合の少なさの重み", 0.0, 1.0, 0.3)
    weight_recency = st.slider("情報の古さの重み", 0.0, 1.0, 0.2)

start_button = st.sidebar.button("🚀 リサーチ＆構成作成スタート", type="primary")

# --- 3. メインロジック ---
if start_button:
    # バリデーションチェック（ScraperAPIキーの確認を追加）
    if not api_key or not scraper_api_key or not keyword:
        st.warning("⚠️ OpenAI APIキー、ScraperAPIキー、リサーチキーワードの3点は必須です。")
        st.stop()

    my_bar = st.progress(0, text="noteの市場データを解析中...")
    
    with st.spinner('noteからデータを取得・スコアリングしています...（約1〜2分）'):
        # 引数に scraper_api_key を追加してデータ取得を実行
        df_raw = fetch_note_data(keyword, scraper_api_key, max_pages)
        
        if df_raw.empty:
            st.warning("対象データが見つかりませんでした、またはアクセスが拒否されました。別のキーワードをお試しください。")
            st.stop()
            
        df_scored = calculate_advanced_score(df_raw, weight_demand, weight_density, weight_recency)
        my_bar.progress(50, text="AIが最適な構成案を生成中...")
        
        final_plan = generate_content_plan(df_scored, target_reader, user_strength, api_key)
        my_bar.progress(100, text="処理完了！")

    # --- 4. 結果表示（タブUI） ---
    st.success("✨ リサーチと構成の作成が完了しました！")
    
    tab1, tab2, tab3 = st.tabs(["🏆 ブルーオーシャン候補", "📝 記事構成＆戦略", "📊 データ出力（CSV）"])
    
    with tab1:
        st.markdown("### 狙い目テーマTOP5")
        st.dataframe(df_scored[['title', 'total_score', 'demand_score', 'density_score', 'recency_score', 'url']].head(5))
        
    with tab2:
        st.markdown("### あなた専用のnote構成案")
        st.markdown(final_plan)
        
        # ダウンロードボタン（構成案）
        st.download_button("📥 構成案をMarkdownでダウンロード", final_plan, file_name="note_plan.md")

    with tab3:
        st.markdown("### 取得した市場データ")
        st.dataframe(df_scored)
        
        # ダウンロードボタン（CSV）
        csv = df_scored.to_csv(index=False).encode('utf-8-sig') # 文字化け防止
        st.download_button(
            label="📊 市場データをCSVでダウンロード", 
            data=csv, 
            file_name="blue_ocean_data.csv", 
            mime="text/csv"
        )
