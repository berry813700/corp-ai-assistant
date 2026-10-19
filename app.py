import streamlit as st
import requests
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 網頁基本設定
st.set_page_config(page_title="企業知識與 IT 支援助手", page_icon="🤖", layout="wide")
st.title("🤖 企業知識與 IT 支援助手")
st.markdown("您可以詢問任何關於 IT 支援（如忘記密碼、VPN設定）或人資規定的問題。")

# 2. 載入本地向量資料庫
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 3. 側邊欄：API Key 設定與新功能「即時學習」
with st.sidebar:
    st.header("⚙️ 系統設定")
    gemini_api_key = st.text_input("請輸入您的 Gemini API Key", type="password")
    st.info("提示：您的 API Key 僅用於本次對話，不會被儲存。")
    
    st.divider()  # 分隔線
    
    # 🌟 新功能：上傳文件即時學習
    st.header("📚 教導 AI 新知識")
    st.markdown("上傳新的 `.txt` 內部文件，讓 AI 立即學會新規範。")
    uploaded_file = st.file_uploader("上傳文件", type=["txt"])
    
    if uploaded_file is not None:
        if st.button("上傳並讓 AI 學習"):
            with st.spinner("🧠 AI 正在閱讀並吸收知識..."):
                try:
                    # 讀取上傳的文字檔內容
                    stringio = uploaded_file.getvalue().decode("utf-8")
                    
                    # 建立文件切塊器 (避免文章太長塞不進大腦)
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    
                    # 建立文件物件並切塊
                    doc = Document(page_content=stringio, metadata={"source": uploaded_file.name})
                    split_docs = text_splitter.split_documents([doc])
                    
                    # 寫入 Chroma 向量資料庫
                    vectorstore.add_documents(split_docs)
                    
                    st.success(f"✅ 學習完成！AI 已將《{uploaded_file.name}》納入大腦。")
                except Exception as e:
                    st.error(f"❌ 學習失敗：{e}")

# --- 輔助函式 ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_gemini(prompt_text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 0.0}
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"API 呼叫失敗，錯誤碼: {response.status_code}\n詳細訊息: {response.text}"
# ----------------

# 4. 聊天紀錄初始化
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是企業專屬 AI 助手。請問您今天需要什麼協助？"}]

# 顯示過去的對話紀錄
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. 處理使用者的輸入
if prompt := st.chat_input("請輸入您的問題..."):
    if not gemini_api_key:
        st.error("⚠️ 請先在左側欄位輸入您的 Gemini API Key！")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    system_prompt = (
        "你是一位專業的企業內部 IT 與知識支援助手。\n"
        "請「僅根據」以下提供的內部文件內容來回答員工的問題，回答請盡量有條理（可使用列點）。\n"
        "如果文件中沒有相關資訊，請直接回答：「內部知識庫目前沒有相關紀錄，請提交 Ticket 聯繫 IT 支援部門（分機 1234）。」\n\n"
        "內部文件內容：\n"
        "{context}\n\n"
        "員工提問：\n"
        "{input}"
    )

    with st.chat_message("assistant"):
        with st.spinner("🔍 正在查詢企業內部知識庫..."):
            docs = retriever.invoke(prompt)
            context_text = format_docs(docs)
            final_prompt = system_prompt.format(context=context_text, input=prompt)
            answer = ask_gemini(final_prompt, gemini_api_key)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})