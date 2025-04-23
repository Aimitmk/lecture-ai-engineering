import os
import io
import base64
import tempfile
import time
import PyPDF2
import openai
import streamlit as st
from tqdm import tqdm
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import re
import json

# ページ設定
st.set_page_config(
    page_title="PDF OCR 抽出ツール (GPT-4o-mini)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトルと説明
st.title("PDF文書からテキスト抽出ツール（GPT-4o-mini使用）")
st.markdown("**PDFファイルをアップロードし、GPT-4o-miniを使用してテキストを抽出します。**")

# サイドバー
st.sidebar.header("設定")
st.sidebar.markdown("PDFからテキストを抽出するための設定を行います。")

# OpenAI APIキーの入力
api_key = st.sidebar.text_input("OpenAI APIキー", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key
    st.sidebar.success("APIキーが設定されました")

# ヘルパー関数
def encode_image_to_base64(image):
    """PIL Imageオブジェクトをbase64エンコードされた文字列に変換する"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_text_with_gpt4o_mini(image, retry_count=3, retry_delay=5):
    """GPT-4o-miniモデルを使用して画像からテキストを抽出する関数"""
    base64_image = encode_image_to_base64(image)

    # APIリクエストのためのプロンプトを設定
    prompt = "この画像に含まれるすべてのテキストを抽出し、元のレイアウトをできるだけ維持してください。段落、箇条書き、表などの構造を保持してください。"

    for attempt in range(retry_count):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "あなたは高精度なOCRシステムです。画像からテキストを正確に抽出してください。"},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=4096
            )

            # 抽出されたテキストを返す
            return response.choices[0].message.content

        except Exception as e:
            if attempt < retry_count - 1:  # 最後の試行以外
                st.warning(f"API呼び出し中にエラーが発生しました: {str(e)}")
                st.info(f"{retry_delay}秒後に再試行します... (試行 {attempt + 1}/{retry_count})")
                time.sleep(retry_delay)
            else:
                st.error(f"APIリクエスト失敗: {str(e)}")
                return f"OCR処理に失敗しました: {str(e)}"

    return "テキスト抽出に失敗しました。"

def process_pdf(pdf_file, batch_size=5, start_page=1, end_page=None, progress_bar=None, status_text=None):
    """PDFファイルからテキストを抽出する関数（バッチ処理対応）"""
    try:
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.getvalue())
            pdf_path = temp_file.name

        # PDFファイルのメタデータを取得
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)

            # ページ範囲の検証と設定
            if start_page < 1:
                start_page = 1
            if end_page is None or end_page > total_pages:
                end_page = total_pages

            st.info(f"PDFファイル: {pdf_file.name}")
            st.info(f"総ページ数: {total_pages}")
            st.info(f"処理範囲: {start_page}〜{end_page}ページ")

        # 処理するページ範囲
        pages_to_process = list(range(start_page - 1, end_page))
        total_pages_to_process = len(pages_to_process)

        # 結果を格納する辞書
        extracted_text = {}

        # バッチ処理（メモリ使用量を抑える）
        for batch_start in range(0, total_pages_to_process, batch_size):
            batch_end = min(batch_start + batch_size, total_pages_to_process)
            batch_pages = pages_to_process[batch_start:batch_end]

            if status_text:
                status_text.text(f"バッチ処理: {batch_start + 1}〜{batch_end}番目のページ（実際のページ番号: {batch_pages[0] + 1}〜{batch_pages[-1] + 1}）")

            try:
                # PDFをページごとに画像に変換
                images = convert_from_path(pdf_path, first_page=batch_pages[0] + 1, last_page=batch_pages[-1] + 1)
            except Exception as pdf_error:
                if "poppler" in str(pdf_error).lower():
                    st.error("エラー: popplerがインストールされていません。")
                    return {}
                else:
                    raise pdf_error

            # 各ページを処理
            for i, image in enumerate(images):
                page_num = batch_pages[i] + 1  # 1から始まるページ番号
                if status_text:
                    status_text.text(f"ページ {page_num}/{end_page} を処理中...")
                
                if progress_bar:
                    progress_value = (batch_start + i) / total_pages_to_process
                    progress_bar.progress(progress_value)

                # GPT-4o-miniによるテキスト抽出
                page_text = extract_text_with_gpt4o_mini(image)
                extracted_text[page_num] = page_text

                # メモリ解放
                del image

            # バッチ間で一時停止して、APIレート制限に引っかからないようにする
            if batch_end < total_pages_to_process:
                if status_text:
                    status_text.text("API制限を回避するため10秒間停止します...")
                time.sleep(10)

        # 一時ファイルの削除
        os.unlink(pdf_path)
        
        if progress_bar:
            progress_bar.progress(1.0)
        if status_text:
            status_text.text("処理完了！")
            
        return extracted_text

    except Exception as e:
        st.error(f"PDFの処理中にエラーが発生しました: {str(e)}")
        # スタックトレースを表示
        import traceback
        st.code(traceback.format_exc())
        return {}

# アプリのメイン部分
def main():
    # APIキーのチェック
    if not api_key:
        st.warning("OpenAI APIキーを入力してください（サイドバーから入力できます）。")
        return

    # PDFファイルのアップロード
    st.header("PDFファイルのアップロード")
    uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type=["pdf"])
    
    if uploaded_file is None:
        st.info("PDFファイルをアップロードしてください。")
        return

    # 処理設定
    st.header("処理設定")
    
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input("バッチサイズ", min_value=1, max_value=20, value=5, 
                                    help="一度に処理するページ数です。メモリ制約がある場合は小さい値を設定してください。")
        start_page = st.number_input("開始ページ", min_value=1, value=1)
    
    with col2:
        # 終了ページはアップロード後に設定
        if uploaded_file:
            # PDFファイルの総ページ数を取得
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            total_pages = len(pdf_reader.pages)
            end_page = st.number_input("終了ページ", min_value=start_page, max_value=total_pages, value=min(total_pages, 5))
            # ファイルポインタをリセット
            uploaded_file.seek(0)
        else:
            end_page = st.number_input("終了ページ", min_value=1, value=5)
        
        save_location = st.selectbox("保存先", ["ダウンロード", "セッション内で表示"], 
                                    help="抽出したテキストの保存方法を選択します。")

    # 処理開始ボタン
    if st.button("テキスト抽出開始"):
        if uploaded_file:
            st.header("処理状況")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 処理時間の計測開始
            start_time = time.time()
            
            # PDFの処理
            status_text.text("PDFの処理を開始します...")
            extracted_text = process_pdf(
                uploaded_file, 
                batch_size=batch_size, 
                start_page=start_page, 
                end_page=end_page,
                progress_bar=progress_bar,
                status_text=status_text
            )
            
            # 処理時間の計測終了
            end_time = time.time()
            processing_time = end_time - start_time
            
            if not extracted_text:
                st.error("テキスト抽出に失敗しました。")
                return
            
            # 処理結果の表示
            st.header("処理結果")
            st.success(f"処理時間: {processing_time:.2f}秒")
            st.success(f"処理したページ数: {len(extracted_text)}")
            
            # 抽出したテキストを結合（ページ番号順）
            sorted_pages = sorted(extracted_text.keys())
            full_text = ""
            for page in sorted_pages:
                full_text += f"\n==== ページ {page} ====\n\n"
                full_text += extracted_text[page] + "\n"
            
            # テキストの保存/表示
            if save_location == "ダウンロード":
                # テキストファイルとしてダウンロード
                base_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
                txt_file_name = f"{base_name}_extracted.txt"
                
                st.download_button(
                    label="テキストファイルをダウンロード",
                    data=full_text,
                    file_name=txt_file_name,
                    mime="text/plain",
                )
            
            # セッション内で表示
            with st.expander("抽出されたテキストを表示", expanded=True):
                # 最初の数ページのサンプル表示
                for page in sorted_pages:
                    st.subheader(f"ページ {page}")
                    st.text_area(f"ページ{page}のテキスト", extracted_text[page], height=200)

if __name__ == "__main__":
    main()
