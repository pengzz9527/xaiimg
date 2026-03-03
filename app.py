import base64
import datetime as dt
import hashlib
import io
import os
from dataclasses import dataclass

import duckdb
import requests
import streamlit as st
from PIL import Image


APP_TITLE = "xAI 图片生成器"
DEFAULT_MODEL = "grok-2-image"
DEFAULT_XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai")  # 可选：区域 endpoint
DB_PATH = os.getenv("DB_PATH", "data/app.duckdb")
IMAGE_DIR = os.getenv("IMAGE_DIR", "generated_images")


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
            help="默认 https://api.x.ai；如需区域 endpoint，可改成 https://us-east-1.api.x.ai 之类。",
        )
        st.session_state["xai_base_url"] = base_url

        model = st.text_input("模型", value=st.session_state.get("xai_model", DEFAULT_MODEL))
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
                    st.error(f"请求失败：{e.response.status_code} {e.response.text[:400]}")
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

            grid_cols = st.columns(min(3, len(last_results)))
            for j, res in enumerate(last_results):
                with grid_cols[j % len(grid_cols)]:
                    try:
                        img = Image.open(io.BytesIO(res.image_bytes))
                        st.image(img, caption=res.filename, use_container_width=True)
                    except Exception:
                        st.image(res.image_bytes, caption=res.filename, use_container_width=True)

                    st.download_button(
                        label="下载图片",
                        data=res.image_bytes,
                        file_name=res.filename,
                        mime=res.image_mime,
                        use_container_width=True,
                    )

            with st.expander("查看 revised_prompt（如果有）"):
                for res in last_results:
                    if res.revised_prompt:
                        st.markdown(f"**{res.filename}**")
                        st.write(res.revised_prompt)

    with col_right:
        st.subheader("历史记录")
        df = list_history(limit=history_limit)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "导出历史 CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="generations_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "提示：如果在 Streamlit Cloud 上部署，服务端存储可能会在重启后丢失；但图片下载到你本地不会受影响。"
        )


if __name__ == "__main__":
    main()
