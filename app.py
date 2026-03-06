import base64
import datetime as dt
import hashlib
import io
import json
import os
from dataclasses import dataclass

import duckdb
import requests
import streamlit as st
from PIL import Image


APP_TITLE = "xAI 图片生成器"
DEFAULT_MODEL = "grok-2-image"  # 若账号无权限，请在界面选择你可用的模型
DEFAULT_XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai")  # 可选：区域 endpoint
DB_PATH = os.getenv("DB_PATH", "data/app.duckdb")
IMAGE_DIR = os.getenv("IMAGE_DIR", "generated_images")
CONFIG_FILE = os.getenv("CONFIG_FILE", "config/telegram_config.json")  # Telegram 配置本地文件路径


@dataclass
class GenResult:
    created_at: dt.datetime
    prompt: str
    revised_prompt: str | None
    model: str
    idx: int
    image_bytes: bytes
    image_mime: str
    filename: str


def init_storage():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)  # 确保配置目录存在

    con = duckdb.connect(DB_PATH)
    con.execute(
        """
        create table if not exists generations (
            id varchar primary key,
            created_at timestamp,
            model varchar,
            prompt varchar,
            revised_prompt varchar,
            n integer,
            response_format varchar,
            image_mime varchar,
            image_filename varchar
        )
        """
    )
    con.close()


def load_telegram_config() -> dict:
    """从本地 JSON 文件加载 Telegram 配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_telegram_config(bot_token: str, chat_id: str):
    """保存 Telegram 配置到本地 JSON 文件（不保存到数据库）"""
    config = {
        "bot_token": bot_token,
        "chat_id": chat_id,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()
    }
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def send_image_to_telegram(bot_token: str, chat_id: str, image_bytes: bytes, filename: str, caption: str = "") -> bool:
    """发送图片到 Telegram"""
    if not bot_token or not chat_id:
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    
    try:
        files = {
            'photo': (filename, io.BytesIO(image_bytes), 'image/jpeg')
        }
        data = {
            'chat_id': chat_id,
            'caption': caption[:1024] if caption else ""  # Telegram  caption 长度限制
        }
        
        response = requests.post(url, files=files, data=data, timeout=60)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"发送到 Telegram 失败: {str(e)}")
        return False


def xai_list_models(base_url: str, api_key: str) -> list[str]:
    base_url = base_url.strip().rstrip("/")
    url = f"{base_url}/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json().get("data", [])
    # OpenAI 兼容：data 是对象数组，id 为模型名
    models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
    return sorted(set(models))


def xai_images_generate(base_url: str, api_key: str, model: str, prompt: str, n: int = 1, response_format: str = "b64_json"):
    base_url = base_url.strip().rstrip("/")
    url = f"{base_url}/v1/images/generations"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    body = {
        "model": model,
        "prompt": prompt,
        "n": int(n),
        "response_format": response_format,
    }

    # 不要打印 headers/body（避免 key 泄漏到日志）
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    return r.json()


def decode_image_from_b64_json(b64_json: str) -> tuple[bytes, str]:
    """Returns (bytes, mime). Supports raw base64 or data URL."""
    if b64_json.startswith("data:"):
        # data:image/png;base64,....
        header, b64data = b64_json.split(",", 1)
        mime = header.split(";")[0].replace("data:", "")
        return base64.b64decode(b64data), mime

    # xAI 文档描述为 b64_json（通常是纯 base64），生成格式一般为 jpg
    raw = base64.b64decode(b64_json)
    return raw, "image/jpeg"


def safe_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:24]


def save_generation_to_duckdb(
    created_at: dt.datetime,
    model: str,
    prompt: str,
    revised_prompt: str | None,
    n: int,
    response_format: str,
    image_mime: str,
    image_filename: str,
):
    con = duckdb.connect(DB_PATH)
    gid = safe_id(created_at.isoformat(), model, prompt, image_filename)
    con.execute(
        """
        insert or replace into generations
        (id, created_at, model, prompt, revised_prompt, n, response_format, image_mime, image_filename)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [gid, created_at, model, prompt, revised_prompt, n, response_format, image_mime, image_filename],
    )
    con.close()


def list_history(limit: int = 50):
    con = duckdb.connect(DB_PATH)
    df = con.execute(
        """
        select created_at, model, prompt, revised_prompt, image_mime, image_filename
        from generations
        order by created_at desc
        limit ?
        """,
        [int(limit)],
    ).df()
    con.close()
    return df


def load_image_bytes(filename: str) -> bytes | None:
    path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_storage()

    st.title(APP_TITLE)
    st.caption("Streamlit + DuckDB + xAI Images API（key 只在界面输入，不落盘）")

    # 加载已保存的 Telegram 配置
    telegram_config = load_telegram_config()

    with st.sidebar:
        st.header("配置")

        # key 只存 session_state
        api_key = st.text_input(
            "xAI API Key",
            type="password",
            value=st.session_state.get("xai_api_key", ""),
            help="仅保存在当前浏览器会话内（session_state），不会写入数据库/文件。",
        )
        st.session_state["xai_api_key"] = api_key

        base_url = st.text_input(
            "API Base URL（可选）",
            value=st.session_state.get("xai_base_url", DEFAULT_XAI_BASE_URL),
            help="默认 https://api.x.ai ；如需区域 endpoint，可改成 https://us-east-1.api.x.ai 之类。",
        )
        st.session_state["xai_base_url"] = base_url

        st.subheader("模型")

        load_models = st.checkbox(
            "自动加载可用模型列表（推荐）",
            value=st.session_state.get("load_models", True),
            help='会调用 /v1/models 获取你这个 Key 可用的模型，避免出现"模型不存在/无权限"的 404。',
        )
        st.session_state["load_models"] = load_models

        models: list[str] = st.session_state.get("available_models", [])
        if api_key and load_models:
            try:
                models = xai_list_models(base_url=base_url, api_key=api_key)
                st.session_state["available_models"] = models
            except Exception:
                # 不阻断主流程：仍允许手动填写模型
                pass

        if models:
            # 默认选：优先包含 image 的模型，否则回退上次选择/DEFAULT
            preferred = st.session_state.get("xai_model", DEFAULT_MODEL)
            default_index = 0
            if preferred in models:
                default_index = models.index(preferred)
            else:
                for k, mid in enumerate(models):
                    if "image" in mid.lower() or "imagine" in mid.lower():
                        default_index = k
                        break

            model = st.selectbox("选择模型", options=models, index=default_index)
        else:
            model = st.text_input("模型（手动输入）", value=st.session_state.get("xai_model", DEFAULT_MODEL))

        st.session_state["xai_model"] = model

        n = st.slider("一次生成张数 n", min_value=1, max_value=10, value=int(st.session_state.get("xai_n", 1)))
        st.session_state["xai_n"] = n

        response_format = st.selectbox(
            "返回格式",
            options=["b64_json", "url"],
            index=0,
            help="为便于在 Streamlit 里展示与下载，推荐 b64_json。url 会给托管链接。",
        )

        history_limit = st.slider("历史记录展示条数", min_value=10, max_value=200, value=50, step=10)

        # ==================== Telegram 配置区域 ====================
        st.divider()
        st.subheader("📱 Telegram 配置")
        st.caption("配置将保存到本地文件，不存入数据库")
        
        # 从本地配置或 session_state 读取默认值
        default_bot_token = st.session_state.get("telegram_bot_token", telegram_config.get("bot_token", ""))
        default_chat_id = st.session_state.get("telegram_chat_id", telegram_config.get("chat_id", ""))
        
        telegram_bot_token = st.text_input(
            "Bot Token",
            type="password",
            value=default_bot_token,
            help="从 @BotFather 获取的 Bot Token",
        )
        st.session_state["telegram_bot_token"] = telegram_bot_token
        
        telegram_chat_id = st.text_input(
            "Chat ID",
            value=default_chat_id,
            help="目标聊天 ID（可以是用户 ID 或频道/群组 ID）",
        )
        st.session_state["telegram_chat_id"] = telegram_chat_id
        
        # 保存配置按钮
        if st.button("💾 保存 Telegram 配置", use_container_width=True):
            if telegram_bot_token and telegram_chat_id:
                save_telegram_config(telegram_bot_token, telegram_chat_id)
                st.success("✅ Telegram 配置已保存到本地文件")
            else:
                st.warning("请填写 Bot Token 和 Chat ID")
        
        # 测试连接按钮
        if st.button("🧪 测试 Telegram 连接", use_container_width=True):
            if telegram_bot_token and telegram_chat_id:
                with st.spinner("发送测试消息..."):
                    test_message = f"🎨 xAI 图片生成器连接测试\n时间: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    test_url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
                    try:
                        response = requests.post(
                            test_url,
                            json={"chat_id": telegram_chat_id, "text": test_message},
                            timeout=30
                        )
                        response.raise_for_status()
                        st.success("✅ 连接成功！请检查 Telegram 是否收到测试消息")
                    except Exception as e:
                        st.error(f"❌ 连接失败: {str(e)}")
            else:
                st.warning("请先填写 Bot Token 和 Chat ID")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("生成")
        prompt = st.text_area(
            "提示词（Prompt）",
            height=140,
            placeholder="例如：一只戴墨镜的柴犬，赛博朋克风格，夜晚霓虹灯街道，电影级光影",
        )

        gen_btn = st.button("生成图片", type="primary", use_container_width=True)

        if gen_btn:
            if not api_key:
                st.error("请先在侧边栏输入 xAI API Key")
                st.stop()
            if not prompt.strip():
                st.error("请先输入提示词")
                st.stop()

            with st.spinner("生成中…"):
                created_at = dt.datetime.now(dt.timezone.utc).astimezone()
                try:
                    resp = xai_images_generate(
                        base_url=base_url,
                        api_key=api_key,
                        model=model.strip(),
                        prompt=prompt.strip(),
                        n=n,
                        response_format=response_format,
                    )
                except requests.HTTPError as e:
                    # 输出简化错误，避免把敏感信息暴露
                    status = getattr(e.response, "status_code", "?")
                    body = (getattr(e.response, "text", "") or "")[:600]
                    st.error(f"请求失败：{status} {body}")
                    if status == 404 and "model" in body.lower():
                        st.info("看起来是模型不可用/无权限。请在侧边栏勾选"自动加载可用模型列表"，然后从下拉框选择你有权限的模型。")
                    st.stop()
                except Exception as e:
                    st.error(f"请求失败：{e}")
                    st.stop()

                data = resp.get("data", [])
                if not data:
                    st.warning("接口没有返回图片数据")
                    st.json(resp)
                    st.stop()

                st.success(f"生成完成：{len(data)} 张")

                results: list[GenResult] = []
                for i, item in enumerate(data, start=1):
                    revised_prompt = item.get("revised_prompt")

                    if response_format == "url":
                        url = item.get("url")
                        if not url:
                            continue
                        # 对 url 模式：下载回来以便展示/下载
                        img_r = requests.get(url, timeout=120)
                        img_r.raise_for_status()
                        img_bytes = img_r.content
                        mime = img_r.headers.get("content-type", "image/jpeg")
                    else:
                        b64_json = item.get("b64_json")
                        if not b64_json:
                            continue
                        img_bytes, mime = decode_image_from_b64_json(b64_json)

                    ext = "jpg"
                    if "png" in mime:
                        ext = "png"
                    elif "webp" in mime:
                        ext = "webp"

                    filename = f"{created_at.strftime('%Y%m%d_%H%M%S')}_{safe_id(prompt, str(i))}.{ext}"
                    out_path = os.path.join(IMAGE_DIR, filename)
                    with open(out_path, "wb") as f:
                        f.write(img_bytes)

                    save_generation_to_duckdb(
                        created_at=created_at,
                        model=model.strip(),
                        prompt=prompt.strip(),
                        revised_prompt=revised_prompt,
                        n=n,
                        response_format=response_format,
                        image_mime=mime,
                        image_filename=filename,
                    )

                    results.append(
                        GenResult(
                            created_at=created_at,
                            prompt=prompt.strip(),
                            revised_prompt=revised_prompt,
                            model=model.strip(),
                            idx=i,
                            image_bytes=img_bytes,
                            image_mime=mime,
                            filename=filename,
                        )
                    )

                st.session_state["last_results"] = results

        # 展示最近一次结果
        last_results: list[GenResult] = st.session_state.get("last_results", [])
        if last_results:
            st.divider()
            st.subheader("最近一次生成")

            # 获取当前 Telegram 配置
            current_bot_token = st.session_state.get("telegram_bot_token", "")
            current_chat_id = st.session_state.get("telegram_chat_id", "")

            grid_cols = st.columns(min(3, len(last_results)))
            for j, res in enumerate(last_results):
                with grid_cols[j % len(grid_cols)]:
                    try:
                        img = Image.open(io.BytesIO(res.image_bytes))
                        st.image(img, caption=res.filename, use_container_width=True)
                    except Exception:
                        st.image(res.image_bytes, caption=res.filename, use_container_width=True)

                    # 下载按钮
                    st.download_button(
                        label="⬇️ 下载图片",
                        data=res.image_bytes,
                        file_name=res.filename,
                        mime=res.image_mime,
                        use_container_width=True,
                        key=f"download_{j}"
                    )

                    # Telegram 发送按钮（仅在配置完整时显示）
                    if current_bot_token and current_chat_id:
                        if st.button(f"📤 发送到 Telegram", use_container_width=True, key=f"tg_send_{j}"):
                            with st.spinner("发送中..."):
                                caption = f"🎨 提示词: {res.prompt}\n🤖 模型: {res.model}\n🕐 {res.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
                                if res.revised_prompt:
                                    caption += f"\n✨ 优化提示词: {res.revised_prompt}"
                                
                                success = send_image_to_telegram(
                                    bot_token=current_bot_token,
                                    chat_id=current_chat_id,
                                    image_bytes=res.image_bytes,
                                    filename=res.filename,
                                    caption=caption
                                )
                                if success:
                                    st.success("✅ 发送成功！")
                    else:
                        st.info("⚙️ 在侧边栏配置 Telegram 后可发送", icon="ℹ️")

            with st.expander("查看 revised_prompt（如果有）"):
                for res in last_results:
                    if res.revised_prompt:
                        st.markdown(f"**{res.filename}**")
                        st.write(res.revised_prompt)


if __name__ == "__main__":
    main()
