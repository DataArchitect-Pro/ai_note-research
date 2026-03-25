import streamlit as st
import pandas as pd
from scraper_scorer import fetch_note_data, calculate_advanced_score
from content_generator import generate_content_plan, generate_market_summary, expand_keywords_with_perplexity

st.set_page_config(page_title="noteブルーオーシャン発掘ツール", layout="wide")

# --- 1. 簡易パスワード認証 ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.markdown("### 🔒 ツールへのログイン")
        pwd = st.text_input("購入者限定パスワードを入力してください", type="password")
        if pwd == "tN2@mlVMg6wQNLRShy": 
            st.session_state["password_correct"] = True
            st.rerun()
        elif pwd:
            st.error("パスワードが間違っています。")
        st.stop()

check_password()

if 'search_done' not in st.session_state:
    st.session_state['search_done'] = False

# --- 2. サイドバー（入力UI） ---
st.sidebar.title("⚙️ 設定・入力")

st.sidebar.markdown("**1. APIキーの設定**")
api_key = st.sidebar.text_input("OpenAI APIキー (sk-...)", type="password", help="構成案の自動生成に使用します")
scraper_api_key = st.sidebar.text_input("ScraperAPIキー", type="password", help="noteのデータ取得（ブロック回避）に使用します")
perplexity_api_key = st.sidebar.text_input("Perplexity APIキー (任意)", type="password", help="入力すると最新のウェブトレンドを加味した高度なリサーチが自動で実行されます")

st.sidebar.markdown("---")
st.sidebar.markdown("**2. リサーチ条件**")
keyword = st.sidebar.text_input("🔍 リサーチキーワード", placeholder="例：副業 エクセル 時短")
target_reader = st.sidebar.text_area("👤 ターゲット読者像", placeholder="例：毎日残業に追われている30代の経理担当者")
user_strength = st.sidebar.text_area("💪 あなたの本業・強み", placeholder="例：メーカー経理歴10年、VBAコードが書ける")

st.sidebar.markdown("---")
with st.sidebar.expander("詳細スコアリング設定"):
    max_pages = st.slider("データ取得量（ページ数）", 1, 5, 2)
    weight_demand = st.slider("需要（スキ数）の重み", 0.0, 1.0, 0.5)
    weight_density = st.slider("競合の少なさの重み", 0.0, 1.0, 0.3)
    weight_recency = st.slider("情報の古さの重み", 0.0, 1.0, 0.2)

start_button = st.sidebar.button("🚀 リサーチ＆構成作成スタート", type="primary")

# --- 3. メインロジック ---
if start_button:
    if not api_key or not scraper_api_key or not keyword:
        st.warning("⚠️ OpenAI APIキー、ScraperAPIキー、リサーチキーワードの3点は必須です。")
        st.stop()

    my_bar = st.progress(0, text="リサーチを開始します...")
    
    with st.spinner('市場データを収集中です...（約1〜3分）'):
        search_keywords = [keyword]
        
        if perplexity_api_key:
            my_bar.progress(10, text="Perplexityで最新トレンドを分析中...")
            expanded = expand_keywords_with_perplexity(keyword, perplexity_api_key)
            if expanded:
                search_keywords.extend(expanded)
                st.info(f"💡 Perplexityが最新トレンドを検知し、リサーチ範囲を拡張しました:\n{', '.join(expanded)}")
        
        all_raw_dfs = []
        total_kws = len(search_keywords)
        for i, kw in enumerate(search_keywords):
            progress_pct = 20 + int(30 * (i / total_kws))
            my_bar.progress(progress_pct, text=f"「{kw}」のデータをnoteから取得中...")
            df = fetch_note_data(kw, scraper_api_key, max_pages)
            if not df.empty:
                all_raw_dfs.append(df)
                
        if not all_raw_dfs:
            st.warning("対象データが見つかりませんでした。別のキーワードをお試しください。")
            st.stop()
            
        df_raw_combined = pd.concat(all_raw_dfs).drop_duplicates(subset=['url']).reset_index(drop=True)
        
        my_bar.progress(60, text="取得データをスコアリング中...")
        df_scored = calculate_advanced_score(df_raw_combined, weight_demand, weight_density, weight_recency)
        
        my_bar.progress(70, text="AIが最適な構成案と市場サマリーを生成中...")
        final_plan = generate_content_plan(df_scored, target_reader, user_strength, api_key)
        
        # 【改修】市場分析にキーワードと強みを引き渡し、文脈のズレを防ぐ
        keywords_str = "、".join(search_keywords)
        market_summary = generate_market_summary(df_scored, api_key, keywords_str, target_reader, user_strength)
        
        st.session_state['df_scored'] = df_scored
        st.session_state['final_plan'] = final_plan
        st.session_state['market_summary'] = market_summary
        st.session_state['search_keywords'] = search_keywords 
        st.session_state['search_done'] = True
        
        my_bar.progress(100, text="処理完了！")

# --- 4. 結果表示 ---
if st.session_state['search_done']:
    df_scored = st.session_state['df_scored']
    final_plan = st.session_state['final_plan']
    market_summary = st.session_state['market_summary']
    used_keywords = st.session_state.get('search_keywords', [])

    st.success("✨ リサーチと構成の作成が完了しました！下にスクロールして結果を確認してください。")
    
    st.markdown("---")
    st.markdown("### 💡 市場データの傾向とAI分析")
    if len(used_keywords) > 1:
        st.caption(f"※Perplexity拡張検索を含む {len(used_keywords)}パターンのキーワードで多角的に分析しました。")
    st.info(market_summary)
    
    st.markdown("---")
    st.markdown("### 📝 あなた専用のnote戦略と執筆プロンプト")
    st.markdown(final_plan)
    
    st.download_button("📥 この戦略とプロンプトをMarkdownでダウンロード", final_plan, file_name="note_plan.md")

    st.markdown("---")
    with st.expander("📊 取得した市場データ（分析の根拠となったブルーオーシャン候補）", expanded=False):
        st.markdown(f"**合計 {len(df_scored)}件** の記事データを分析しました。")
        st.dataframe(df_scored[['title', 'total_score', 'demand_score', 'density_score', 'recency_score', 'url']].head(10))
        
        csv = df_scored.to_csv(index=False).encode('utf-8-sig') 
        st.download_button(
            label="📊 全ての市場データをCSVでダウンロード", 
            data=csv, 
            file_name="blue_ocean_data.csv", 
            mime="text/csv"
        )
