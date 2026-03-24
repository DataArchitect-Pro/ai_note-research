def fetch_note_data(keyword: str, max_pages: int = 2) -> pd.DataFrame:
    """noteの検索エンドポイントからデータを取得する（v3対応版）"""
    # 変更点1: v2からv3エンドポイントに変更
    api_url = "https://note.com/api/v3/searches"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    
    all_articles = []
    size = 20 # 1ページあたりの取得件数
    
    for page in range(max_pages):
        # 変更点2: v3用のパラメータ（ページ数ではなく offset 方式の 'start' を使用）
        params = {
            "q": keyword,
            "context": "note",
            "size": size,
            "start": page * size
        }
        try:
            response = requests.get(api_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # 変更点3: v3 APIのより深いJSON階層に合わせてデータを抽出
            response_data = data.get("data", {})
            
            # APIのJSON構造のゆらぎに対応できる堅牢な取得
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
                # ユーザー情報の取得（キーが存在しない場合のエラーを防ぐ）
                user_info = item.get("user", {})
                urlname = user_info.get("urlname", "unknown")
                key = item.get("key", "")
                
                article = {
                    "title": item.get("name"),
                    "like_count": item.get("likeCount", 0),
                    "author": urlname,
                    # createdAtがない場合はpublishAtでフォールバック
                    "created_at": item.get("createdAt", item.get("publishAt")),
                    "url": f"https://note.com/{urlname}/n/{key}" if key else ""
                }
                all_articles.append(article)
                
            time.sleep(random.uniform(1.5, 3.0)) # IPバン対策のウェイト
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                st.error("⚠️ アクセスが集中しているため、note側で一時的な制限がかかりました。数分後に再度お試しください。")
            elif e.response.status_code == 404:
                st.error("⚠️ noteのAPIが見つかりません。APIの仕様がさらに変更された可能性があります。")
            else:
                st.error(f"⚠️ データ取得エラー: {e}")
            break
        except Exception as e:
            st.error(f"⚠️ 予期せぬエラーが発生しました: {e}")
            break
            
    return pd.DataFrame(all_articles)
