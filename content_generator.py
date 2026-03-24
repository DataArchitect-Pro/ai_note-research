import pandas as pd
from openai import OpenAI

def generate_content_plan(df_top: pd.DataFrame, target_reader: str, user_strength: str, api_key: str) -> str:
    """OpenAI APIを使用して販売戦略と全テーマの構成案を生成する"""
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

    system_prompt = "あなたはプロのコンテンツマーケターであり、note専門の凄腕編集者です。データに基づき、読者の心を動かし競合と差別化できる連載・複数記事の構成案をMarkdown形式で作成してください。"

    # 【改善点】すべてのテーマに対して「売れる理由・タイトル・構成」をセットで出させる階層構造に変更
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
    ## 🎯 1. 全体的な商品アイデアと販売戦略
    - **全体の商品アイデア概要**: （どのような内容を誰に提供するのか全体像）

    ## 📝 2. 具体的な執筆テーマと記事構成案（5〜10個）
    今回の切り口を具体化し、連載や複数記事として展開できるテーマを5〜10個提案してください。
    **【重要】必ず提案するすべてのテーマ（5〜10個すべて）に対して、以下のフォーマットで「売れる理由」「タイトル案」「詳細な構成案」を出力してください。**省略は絶対にしないでください。

    ---
    ### 【テーマ1】: [具体的なテーマ名や切り口]
    - **なぜ今の市場で売れるのか**: [ターゲットの深い悩みと、市場の空き状況からの考察]
    - **既存記事との差別化ポイント**: [執筆者の強みをどう活かして勝つか]
    
    **💡 惹きつけるタイトル案（3パターン）**
    1. [クリックしたくなるキャッチーなタイトル]
    2. [ターゲットの悩みに寄り添うタイトル]
    3. [執筆者の権威性・実績を押し出したタイトル]

    **📝 記事の構成案（目次）**
    - **導入**: [読者の悩みへの共感ポイント] / [提示する解決策]
    - **見出し1（H2）: [具体的な見出し名]**
      - [H3]: [ここで語るべき執筆者の一次情報・実体験]
      - [H3]: [読者の疑問への先回り回答]
    - **見出し2（H2）: [具体的な見出し名]**
      - [H3]: [具体的なステップや事例]
      - [H3]: [よくある失敗とその回避法]
    - **まとめと次のアクション**: [読者に促す具体的な行動]

    ---
    ### 【テーマ2】: [具体的なテーマ名や切り口]
    （※テーマ1と全く同じフォーマットで、理由・タイトル3案・構成案をすべて出力）

    ---
    （※以降、テーマ5〜10まで上記と全く同じフォーマットで繰り返し、すべて出力してください）
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            # 出力量が飛躍的に増えるため、最大トークン数を4000に拡張
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AIによる生成エラー: {e}"

def generate_market_summary(df_top: pd.DataFrame, api_key: str) -> str:
    """取得したトップデータから「なぜブルーオーシャンなのか」を解説するサマリーを生成"""
    if df_top.empty:
        return "データがありません。"

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return ""

    top_titles = "\n".join([f"- {row['title']}" for _, row in df_top.head(5).iterrows()])

    prompt = f"""
    あなたはデータアナリストです。以下のnoteで取得した「ブルーオーシャン（需要が高く競合が弱い）と判定された上位5記事」のタイトルを見て、以下の2点を200〜300文字程度のMarkdown形式で簡潔に解説してください。
    
    1. 取得したタイトルの全体的な概要・傾向
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
