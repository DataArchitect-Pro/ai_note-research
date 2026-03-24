import pandas as pd
from openai import OpenAI

def generate_content_plan(df_top: pd.DataFrame, target_reader: str, user_strength: str, api_key: str) -> str:
    """OpenAI APIを使用して構成案を生成する"""
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
        top_3_context += f"- 既存記事{index+1}: 「{row.get('title', '')}」 (スキ数: {row.get('like_count', 0)})\n"

    system_prompt = "あなたはプロのコンテンツマーケターであり、note専門の凄腕編集者です。データに基づき、読者の心を動かし競合と差別化できる記事の構成案をMarkdown形式で作成してください。"

    user_prompt = f"""
    以下の【市場データ】と【執筆者の情報】をもとに、高品質な記事の構成案を作成してください。

    ### 【市場データ（note内のブルーオーシャン領域）】
    今回狙うトップテーマの既存記事タイトル: 「{best_title}」
    競合上位の傾向:
    {top_3_context}
    
    ### 【執筆者の情報】
    ターゲット読者像: {target_reader}
    執筆者の本業・強み: {user_strength}

    ### 【出力フォーマット（厳守）】
    ## 🎯 この記事の戦略とポジショニング
    （なぜこの切り口なら勝てるのか、編集者としての視点を解説）

    ## 💡 惹きつけるタイトル案（3パターン）
    1. [クリックしたくなるタイトル]
    2. [悩みに寄り添うタイトル]
    3. [権威性を押し出したタイトル]

    ## 📝 記事の構成案（目次）
    ### 導入（アイキャッチと共感）
    - 共感ポイント:
    - 解決策の提示:

    ### 見出し1（H2）: [具体的な見出し名]
    - [見出し3（H3）]: [ここで語るべき執筆者の一次情報]
    - [見出し3（H3）]: [読者の疑問への先回り回答]

    ### 見出し2（H2）: [具体的な見出し名]
    - [見出し3（H3）]: [具体的なステップや事例]

    ### まとめと次のアクション
    - 読者に促す具体的な行動:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AIによる生成エラー: {e}"