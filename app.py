import streamlit as st
import pandas as pd
import time
import uuid
from scraper_scorer import fetch_note_data, calculate_advanced_score
from content_generator import generate_content_plan, generate_market_summary, expand_keywords_with_perplexity

st.set_page_config(page_title="noteブルーオーシャン発掘ツール", layout="wide")

# ==========================================
# 1. 認証ロジック (先勝ちブロック + タイムアウト機能)
# ==========================================
@st.cache_resource
def get_active_sessions():
    # 構造: { "user_id": {"token": "...", "last_active": 1690000000.0} }
    return {}

def check_password():
    common_password = st.secrets.get("APP_PASSWORD", "tN2@mlVMg6wQNLRShy")
    allowed_ids = st.secrets.get("ALLOWED_IDS", ["a380.rolls.royce@gmail.com"])
    
    active_sessions = get_active_sessions()
    current_time = time.time()
    
    # 安全装置：30分（1800秒）操作がなかったセッションは「ログアウト忘れ」とみなし、ロックを解除する
    TIMEOUT_SECONDS = 1800 
    for uid in list(active_sessions.keys()):
        if current_time - active_sessions[uid]["last_active"] > TIMEOUT_SECONDS:
            del active_sessions[uid]

    # セッションステートの初期化
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
    if "session_token" not in st.session_state:
        st.session_state["session_token"] = None

    current_user = st.session_state["user_id"]
    current_token = st.session_state["session_token"]

    # 現在ログイン中のユーザーが操作した際の生存確認（タイムアウトの更新）
    if current_user:
        if current_user in active_sessions and active_sessions[current_user]["token"] == current_token:
            # 操作するたびに寿命をリセット（延長）する
            active_sessions[current_user]["last_active"] = current_time
        else:
            # タイムアウト等でサーバーから消去された場合はログアウト状態に戻す
            st.session_state["user_id"] = None
            st.session_state["session_token"] = None
            current_user = None

    # ログインしていない場合の画面表示
    if not current_user:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #333; font-size: 2.5em;'>🔒 ユーザーログイン</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; font-size: 1.3em; margin-bottom: 10px;'>付与された専用IDと、共通パスワードを入力してください。</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #d32f2f; font-size: 0.9em; margin-bottom: 30px;'>※同時ログイン不可（別の人が使用中のIDではログインできません）</p>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                st.markdown("<div style='font-size: 1.0em; font-weight: bold; margin-bottom: 4px; color: #333;'>📝 専用ユーザーID</div>", unsafe_allow_html=True)
                user_id = st.text_input("ID", placeholder="例：noteの注文IDなど", label_visibility="collapsed")
                
                st.markdown("<div style='font-size: 1.0em; font-weight: bold; margin-top: 12px; margin-bottom: 4px; color: #333;'>🔑 共通パスワード</div>", unsafe_allow_html=True)
                password = st.text_input("パスワード", type="password", placeholder="記事内にあるパスワードを入力", label_visibility="collapsed")
                
                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("ログイン", use_container_width=True)

                if submit:
                    if password != common_password:
                        st.error("❌ パスワードが間違っています。")
                    elif user_id not in allowed_ids:
                        st.error("❌ 登録されていないユーザーIDです。")
                    elif user_id in active_sessions:
                        # ここで「後からのログイン」を完全にブロックする
                        st.error("❌ このIDは現在別の端末・ブラウザで利用中です。（同時ログイン不可）\n\n※前の利用者がログアウトするか、一定時間（最大30分）経過するまでお待ちください。")
                    else:
                        # 認証成功：新しいユニークトークンを発行し、時間を記録
                        new_token = str(uuid.uuid4())
                        st.session_state["user_id"] = user_id
                        st.session_state["session_token"] = new_token
                        active_sessions[user_id] = {"token": new_token, "last_active": current_time}
                        st.rerun() 
        
        # 認証されるまで以降のコードを一切実行しない
        st.stop()

# アプリ起動時に必ずパスワードとセッションをチェック
check_password()

if 'search_done' not in st.session_state:
    st.session_state['search_done'] = False


# --- 2. サイドバー（入力UIとログアウト） ---
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

# 【追加】ログアウト機能（IDロックの即時解除）
st.sidebar.markdown("---")
if st.sidebar.button("🚪 ログアウトしてIDを解放する"):
    active_sessions = get_active_sessions()
    user_id = st.session_state.get("user_id")
    if user_id in active_sessions:
        del active_sessions[user_id]
    st.session_state["user_id"] = None
    st.session_state["session_token"] = None
    st.rerun()

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
        
        my_bar.progress(60, text="取得データをスコアリング中（ノイズ除去フィルター適用）...")
        
        user_context_str = f"{keyword} {target_reader} {user_strength}"
        df_scored = calculate_advanced_score(df_raw_combined, weight_demand, weight_density, weight_recency, user_context=user_context_str)
        
        my_bar.progress(70, text="AIが最適な構成案と市場サマリーを生成中...")
        
        keywords_str = "、".join(search_keywords)
        final_plan = generate_content_plan(df_scored, keywords_str, target_reader, user_strength, api_key)
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
        st.markdown(f"**合計 {len(df_scored)}件** の記事データを分析しました。（※関連度スコアが低いノイズ記事は下位に排除されています）")
        st.dataframe(df_scored[['title', 'total_score', 'relevance_score', 'demand_score', 'density_score', 'recency_score', 'url']].head(10))
        
        csv = df_scored.to_csv(index=False).encode('utf-8-sig') 
        st.download_button(
            label="📊 全ての市場データをCSVでダウンロード", 
            data=csv, 
            file_name="blue_ocean_data.csv", 
            mime="text/csv"
        )
