import streamlit as st
import pandas as pd
import time
from scraper_scorer import fetch_note_data, calculate_advanced_score
from content_generator import generate_content_plan

st.set_page_config(page_title="noteブルーオーシャン発掘ツール", layout="wide")

# --- パスワード認証 ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.markdown("### 🔒 ツールへのログイン")
        pwd = st.text_input("購入者限定パスワードを入力してください", type="password")
        if pwd == "note-ai-2026": # 運用時に変更
            st.session_state["password_correct"] = True
            st.rerun()
        elif pwd:
            st.error("パスワードが間違っています。")
        st.stop()

check_password()

# --- サイドバーUI ---
st.sidebar.title("⚙️ 設定・入力")
api_key = st.sidebar.text_input("OpenAI APIキー (sk-...)", type="password")

st.sidebar.markdown("---")
keyword = st.sidebar.text_input("🔍 リサーチキーワード", placeholder="例：経理 エクセル 時短")
target_reader = st.sidebar.text_area("👤 ターゲット読者像", placeholder="例：毎月残業している30代経理")
user_strength = st.sidebar.text_area("💪 あなたの強み", placeholder="例：メーカー経理歴10年、VBAが書ける")

st.sidebar.markdown("---")
with st.sidebar.expander("詳細スコアリング設定"):
    max_pages = st.slider("データ取得量（ページ数）", 1, 5, 2)
    weight_demand = st.slider("需要の重み", 0.0, 1.0, 0.5)
    weight_density = st.slider("競合の少なさの重み", 0.0, 1.0, 0.3)
    weight_recency = st.slider("情報の古さの重み", 0.0, 1.0, 0.2)

start_button = st.sidebar.button("🚀 リサーチ＆構成作成スタート", type="primary")

# --- メインロジック ---
if start_button:
    if not api_key or not keyword:
        st.warning("⚠️ APIキーとキーワードは必須です。")
        st.stop()

    my_bar = st.progress(0, text="noteの市場データを解析中...")
    
    with st.spinner('データを取得・スコアリングしています...'):
        df_raw = fetch_note_data(keyword, max_pages)
        if df_raw.empty:
            st.warning("対象データが見つかりませんでした。別のキーワードをお試しください。")
            st.stop()
            
        df_scored = calculate_advanced_score(df_raw, weight_demand, weight_density, weight_recency)
        my_bar.progress(50, text="AIが最適な構成案を生成中...")
        
        final_plan = generate_content_plan(df_scored, target_reader, user_strength, api_key)
        my_bar.progress(100, text="処理完了！")

    st.success("✨ リサーチと構成の作成が完了しました！")
    
    tab1, tab2, tab3 = st.tabs(["🏆 ブルーオーシャン候補", "📝 記事構成＆戦略", "📊 データ出力（CSV）"])
    
    with tab1:
        st.markdown("### 狙い目テーマTOP5")
        st.dataframe(df_scored[['title', 'total_score', 'demand_score', 'density_score', 'recency_score', 'url']].head(5))
        
    with tab2:
        st.markdown("### あなた専用のnote構成案")
        st.markdown(final_plan)
        st.download_button("📥 構成案をMarkdownでダウンロード", final_plan, file_name="note_plan.md")

    with tab3:
        st.markdown("### 取得した市場データ")
        st.dataframe(df_scored)
        csv = df_scored.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📊 市場データをCSVでダウンロード", data=csv, file_name="blue_ocean_data.csv", mime="text/csv")