import pandas as pd
from openai import OpenAI
import requests

def generate_content_plan(df_top: pd.DataFrame, target_reader: str, user_strength: str, api_key: str) -> str:
    """OpenAI APIを使用して販売戦略と各テーマごとの専用プロンプトを生成する"""
    if df_top.empty:
        return "有効なデータがありません。検索キーワードを変更してお試しください。"

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return f"APIキーの設定エラー: {e}"

    best_theme = df_top.iloc[0]
    best_title = best_theme.get('title', '不明')
    
    top_3_context = ""
    for index, row in df_top.head(3).iterrows():
        top_3_context += f"- 既存記事{index+1}: 「{row.get('title', '')}」 (需要スコア: {row.get('demand_score', 0)})\n"

    system_prompt = """あなたはプロのコンテンツマーケターであり、凄腕のプロンプトエンジニアです。
データに基づき、noteの連載テーマ案と、それを別のAIに書かせるための「専用プロンプト（指示書）」を作成するのがあなたの仕事です。
【厳守事項】あなた自身が記事の目次（H2やH3など）や構成案を出力することは固く禁じます。必ず指定されたプロンプトのフォーマットのみを出力してください。"""

    user_prompt = f"""
    以下の【市場データ】と【執筆者の情報】をもとに、noteの連載テーマ案（厳選5個）と、各テーマの執筆用プロンプトを作成してください。

    ### 【市場データ（note内のブルーオーシャン領域）】
    今回狙うトップテーマの既存記事タイトル: 「{best_title}」
    競合上位の傾向:
    {top_3_context}
    
    ### 【執筆者の情報】
    ターゲット読者像: {target_reader}
    執筆者の本業・強み: {user_strength}

    ### 【出力フォーマット（厳守）】
    ## 🎯 1. 全体的な商品アイデアと販売戦略
    - **全体の商品アイデア概要**: （どのような内容を誰に提供するのか全体像）

    ## 📝 2. 具体的な執筆テーマとAI執筆プロンプト（厳選5テーマ）
    連載や複数記事として展開できるテーマを【必ず5個】提案してください。
    【重要】各テーマについて、以下のフォーマットを「一言一句変えずに」使用して出力してください。目次や構成案は絶対に書かないでください。

    ---
    ### 【テーマ1】: [具体的なテーマ名や切り口]
    - **なぜ今の市場で売れるのか**: [ターゲットの深い悩みと、市場の空き状況からの考察]
    - **既存記事との差別化ポイント**: [執筆者の強みをどう活かして勝つか]
    
    **💡 惹きつけるタイトル案（3パターン）**
    1. [クリックしたくなるキャッチーなタイトル]
    2. [ターゲットの悩みに寄り添うタイトル]
    3. [執筆者の権威性・実績を押し出したタイトル]

    **🤖 この記事をAIに書かせるための専用プロンプト**
    ```text
    あなたはプロのnoteライターです。以下の条件で、読者の心を動かし、購入へ繋がる有料note記事の構成案と本文を作成してください。
    
    【執筆条件】
    ・テーマ：[ここに上記の具体的なテーマ名を記載]
    ・ターゲット読者：{target_reader}
    ・既存記事との差別化：[ここに上記の「既存記事との差別化ポイント」の内容を具体的に記載]
    ・盛り込むべき筆者の強み（一次情報）：{user_strength}
    ・記事のゴール：読者の悩みを解決し、有料部分の購入へ促すこと
    
    【出力指示】
    1. まず、導入〜見出し(H2, H3)〜まとめの「構成案」を作成してください。
    2. 構成案の中で、読者の興味を最大限に惹きつけた上で「ここから先は有料（コアノウハウ）」となる最適な位置に、「---（ここから有料エリア）---」と境界線を明記してください。
    3. その構成案に従って、無料エリア（読者への共感と問題提起）と有料エリア（具体的な手順、差別化された独自のノウハウ、実例）に分けて「本文」を執筆してください。
    ```
    ---
    ### 【テーマ2】: [具体的なテーマ名や切り口]
    （※テーマ1と全く同じフォーマットで出力。目次は書かない。）

    ---
    （※以降、テーマ5まで上記と全く同じフォーマットで繰り返し、必ず5個すべて出力して完了してください）
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AIによる生成エラー: {e}"

# 【改修】引数にキーワード、読者像、強みを追加し、分析の精度を向上
def generate_market_summary(df_top: pd.DataFrame, api_key: str, keywords: str, target_reader: str, user_strength: str) -> str:
    """取得したトップデータから「なぜブルーオーシャンなのか」を解説するサマリーを生成"""
    if df_top.empty:
        return "データがありません。"

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return ""

    top_titles = "\n".join([f"- {row['title']}" for _, row in df_top.head(5).iterrows()])

    # 【改修】ノイズを無視し、入力された文脈に沿って分析するよう強力に指示
    prompt = f"""
    あなたはデータアナリストです。
    ユーザーは「{keywords}」というキーワードで市場調査を行いました。
    ターゲット層は「{target_reader}」、執筆者の強みは「{user_strength}」です。

    以下のnoteで取得した「ブルーオーシャン候補の上位5記事」のタイトルを見て、ユーザーの検索意図に沿った形で、以下の2点を200〜300文字程度のMarkdown形式で簡潔に解説してください。

    【重要なお願い】
    noteの検索エンジンの仕様上、上位記事の中に「ハンドメイド」や「子育て」「無関係な資格」など、今回のキーワード（{keywords}）や強み（{user_strength}）と全く無関係なノイズ記事が混ざっている場合があります。
    その場合は無関係な要素を完全に無視し、あくまで「{keywords}」やユーザーの強みに関連する文脈だけを抽出して、なぜこれがブルーオーシャンなのかを推察してください。

    1. 取得したタイトルの全体的な概要・傾向（※キーワードに関連する部分のみ）
    2. なぜこのテーマが「需要があるのに競合が弱い（または古い）ブルーオーシャン」と言えるのかの推察

    【抽出された上位記事】
    {top_titles}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception:
        return "市場データの解析サマリーの生成に失敗しました。"

def expand_keywords_with_perplexity(keyword: str, perplexity_api_key: str) -> list:
    """Perplexity APIを使用して、ウェブ上の最新トレンドから掛け合わせキーワードを3つ抽出する"""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {perplexity_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system", 
                "content": "あなたは優秀なリサーチャーです。ユーザーのキーワードに関連する、現在ウェブ上で検索されている最新の悩みやトレンドを含む「掛け合わせキーワード（2〜3語の組み合わせ）」を3つ提案してください。出力は絶対にキーワードのみをカンマ区切り（例: 経理 パワークエリ 自動化, 経理 マクロ エラー, インボイス エクセル 突合）で出力し、他の説明や記号は一切含めないでください。"
            },
            {
                "role": "user", 
                "content": f"キーワード: {keyword}"
            }
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        content = data['choices'][0]['message']['content']
        keywords = [k.strip() for k in content.split(',')]
        return [k for k in keywords if k]
    except Exception:
        return []
