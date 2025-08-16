# app.py  —— 含 Magic Link 登录 / 免费3次 / Paddle 的完整版本
import os
import re
import hmac
import json
import time
import secrets
import sqlite3
import hashlib
import tempfile
from datetime import datetime
from urllib.parse import quote_plus

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import gradio as gr
import requests as _rq

from utils import analyze_repo, generate_guide, troubleshoot

# =========================
# 环境变量 / 常量
# =========================
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:7860").rstrip("/")
DB_PATH = os.getenv("DB_PATH", "./app.db")

# 登录 & 会话
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_ME_IN_PROD")  # 生产必须改成随机长串
SESSION_COOKIE = "repoai_session"

# Paddle（可先留空，等你开支付）
PADDLE_CLIENT_TOKEN = os.getenv("PADDLE_CLIENT_TOKEN", "").strip()  # test_... / live_...
PADDLE_PRICE_BASIC  = os.getenv("PADDLE_PRICE_BASIC", "").strip()   # pri_...
PADDLE_PRICE_PRO    = os.getenv("PADDLE_PRICE_PRO", "").strip()     # pri_...
PADDLE_WEBHOOK_SECRET = os.getenv("PADDLE_WEBHOOK_SECRET", "").strip()

PLAN_CREDIT_MAP = {
    "BASIC": {"price_id": PADDLE_PRICE_BASIC, "credits": 200},
    "PRO":   {"price_id": PADDLE_PRICE_PRO,   "credits": 1000},
}
PRICEID_TO_PLAN = {v["price_id"]: (k, v["credits"]) for k, v in PLAN_CREDIT_MAP.items() if v["price_id"]}

# 配额/扣次
CREDITS_COST_PER_RUN = 1   # 每次“开始分析”扣 1
TRIAL_CREDITS = 3          # 试用次数 = 3

# =========================
# 主题 / CSS
# =========================
theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet", neutral_hue="slate")
CSS = """
#header h1 {margin-bottom: 4px;}
#header p {margin-top: 0; color: #6b7280;}
.small-note {font-size: 12px; color: #64748b}
.wrap-code pre {white-space: pre-wrap}
footer {text-align:center; font-size:12px; color:#94a3b8; margin-top:6px}
#checklist {max-height: 520px; overflow-y: auto; padding-right: 8px;}
#checklist label {margin-bottom: 6px}
#right_pane {max-height: calc(100vh - 220px); overflow-y: auto;}
#checklist, #checklist > div, #checklist > div > div {max-height: 520px; overflow-y: auto;}
"""

# =========================
# SQLite
# =========================
import sqlite3
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def init_db():
    conn = db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users(
        email TEXT PRIMARY KEY,
        credits INTEGER DEFAULT 0,
        trial_used INTEGER DEFAULT 0,
        created_at TEXT,
        updated_at TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS events(
        event_id TEXT PRIMARY KEY,
        created_at TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS auth_tokens(
        token TEXT PRIMARY KEY,
        email TEXT,
        expires_at INTEGER,
        used INTEGER DEFAULT 0
    )""")
    conn.commit(); conn.close()

def get_user(email: str):
    conn = db()
    cur = conn.execute("SELECT * FROM users WHERE email=?", (email,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def upsert_user(email: str):
    u = get_user(email)
    conn = db()
    if u is None:
        conn.execute(
            "INSERT INTO users(email, credits, trial_used, created_at, updated_at) VALUES(?,?,?,?,?)",
            (email, 0, 0, now_iso(), now_iso())
        )
    else:
        conn.execute("UPDATE users SET updated_at=? WHERE email=?", (now_iso(), email))
    conn.commit(); conn.close()
    return get_user(email)

def add_credits(email: str, delta: int):
    conn = db()
    conn.execute("UPDATE users SET credits=COALESCE(credits,0)+?, updated_at=? WHERE email=?",
                 (delta, now_iso(), email))
    conn.commit(); conn.close()

def set_trial_used(email: str):
    conn = db()
    conn.execute("UPDATE users SET trial_used=1, updated_at=? WHERE email=?", (now_iso(), email))
    conn.commit(); conn.close()

def mark_event_processed(event_id: str) -> bool:
    conn = db()
    try:
        conn.execute("INSERT INTO events(event_id, created_at) VALUES(?,?)", (event_id, now_iso()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# =========================
# Magic Link 登录工具
# =========================
def _sign(value: str) -> str:
    sig = hmac.new(SECRET_KEY.encode(), value.encode(), hashlib.sha256).hexdigest()
    return f"{value}.{sig}"

def _unsign(signed: str | None) -> str | None:
    if not signed:
        return None
    try:
        value, sig = signed.rsplit(".", 1)
        expect = hmac.new(SECRET_KEY.encode(), value.encode(), hashlib.sha256).hexdigest()
        return value if hmac.compare_digest(sig, expect) else None
    except Exception:
        return None

def create_magic_token(email: str, ttl_seconds: int = 15 * 60) -> str:
    token = secrets.token_urlsafe(32)
    expires = int(time.time()) + ttl_seconds
    conn = db()
    conn.execute(
        "INSERT OR REPLACE INTO auth_tokens(token, email, expires_at, used) VALUES(?,?,?,0)",
        (token, email, expires)
    )
    conn.commit(); conn.close()
    return token

def use_magic_token(token: str) -> str | None:
    conn = db()
    cur = conn.execute("SELECT email, expires_at, used FROM auth_tokens WHERE token=?", (token,))
    row = cur.fetchone()
    if not row:
        conn.close(); return None
    email, exp, used = row["email"], row["expires_at"], row["used"]
    if used or time.time() > exp:
        conn.close(); return None
    conn.execute("UPDATE auth_tokens SET used=1 WHERE token=?", (token,))
    conn.commit(); conn.close()
    return email

# =========================
# 应用状态 & 工具
# =========================
def init_state():
    return {
        "repo_data": None,
        "history": [],
        "user_env": "Windows",
        "steps": [],
        "titles": [],
        "codes": [],
        "warnings": [],
        "user_email": "",
        "user_credits": 0,
    }

_DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+\/\s*$",
    r"\brm\s+-rf\s+\*",
    r"\bmkfs\.|\bmkfs\b",
    r"\bdd\s+if=",
    r"\bformat\s+[A-Za-z]:",
    r"\bdel\s+\/s\s+\/q\s+[A-Za-z]:\\",
]

def extract_code_snippets(text: str):
    if not text:
        return []
    fenced = re.findall(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```", text)
    inline = re.findall(r"`([^`]+)`", text)
    line_cmds = []
    for line in text.splitlines():
        if re.match(r"^(git|pip|python|python3|conda|uvicorn|streamlit|poetry|npm|node|docker|pwsh|powershell|bash|sh|make|uv)\b", line.strip()):
            line_cmds.append(line.strip())
    merged, seen = [], set()
    for seg in fenced + inline + line_cmds:
        seg = seg.strip()
        if not seg or seg in seen:
            continue
        seen.add(seg)
        merged.append(seg)
    return merged

def sanitize_commands(cmds):
    safe, warns = [], []
    for c in cmds:
        if any(re.search(p, c, flags=re.IGNORECASE) for p in _DANGEROUS_PATTERNS):
            warns.append(f"已过滤可疑命令：`{c}`")
        else:
            safe.append(c)
    return safe, warns

def split_steps(text: str):
    if not text:
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(r"(?m)^\s*(\d{1,3})([\.、\)])\s+")
    idx = [m.start() for m in pattern.finditer(normalized)]
    if len(idx) >= 2:
        out = []
        for i, start in enumerate(idx):
            end = idx[i+1] if i+1 < len(idx) else len(normalized)
            out.append(normalized[start:end].strip())
        return [p for p in out if p]
    fallback = re.findall(r"\d+\.[\s\S]*?(?=\n\d+\. |\n\d+\.|$)", normalized)
    return [s.strip() for s in (fallback or [normalized.strip()]) if s.strip()]

# ============ 导出 ============
import tempfile
def export_markdown(state):
    steps = state.get("steps") or []
    repo = (state.get("repo_data") or {}).get("url", "")
    env  = state.get("user_env", "Windows")
    codes = state.get("codes") or []
    md = [f"# 部署指南\n\n- 仓库：{repo}\n- 环境：{env}\n- 生成时间：{datetime.utcnow().isoformat()}Z\n", "## 步骤\n"]
    for i, s in enumerate(steps):
        md.append(s if s.startswith(f"{i+1}. ") else f"{i+1}. {s}")
    if codes:
        md.append("\n## 命令清单\n")
        lang = "powershell" if env == "Windows" else "bash"
        for c in codes:
            md.append(f"```{lang}\n{c}\n```")
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
    f.write("\n\n".join(md).encode("utf-8")); f.flush(); f.close()
    return gr.update(value=f.name, visible=True)

def export_script(state):
    env = state.get("user_env", "Windows")
    codes = state.get("codes") or []
    if not codes:
        codes = (['Write-Host "未提取到命令，请在界面中复制具体命令执行。"']
                 if env == "Windows" else ["echo 未提取到命令，请在界面中复制具体命令执行。"])
    if env == "Windows":
        content = "\n".join(["$ErrorActionPreference = 'Stop'", "Set-StrictMode -Version Latest", *codes])
        suffix = ".ps1"
    else:
        content = "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", *codes])
        suffix = ".sh"
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(content.encode("utf-8")); f.flush(); f.close()
    return gr.update(value=f.name, visible=True)

def export_json(state):
    payload = {
        "repo": (state.get("repo_data") or {}).get("url"),
        "env": state.get("user_env"),
        "steps": state.get("steps") or [],
        "commands": state.get("codes") or [],
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "version": "mvp-1.3",
    }
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    f.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")); f.flush(); f.close()
    return gr.update(value=f.name, visible=True)

# =========================
# Paddle Webhook 验签
# =========================
def verify_paddle_signature(signature_header: str, raw_body: bytes) -> bool:
    if not signature_header or not PADDLE_WEBHOOK_SECRET:
        return False
    try:
        parts = dict(kv.split("=", 1) for kv in signature_header.split(";"))
        ts = parts.get("ts"); h1 = parts.get("h1")
        if not ts or not h1:
            return False
        signed = f"{ts}:{raw_body.decode('utf-8')}".encode("utf-8")
        digest = hmac.new(PADDLE_WEBHOOK_SECRET.encode("utf-8"), signed, hashlib.sha256).hexdigest()
        return hmac.compare_digest(digest, h1)
    except Exception:
        return False

# =========================
# 业务回调
# =========================
def start_analysis(url, env, state):
    email = (state or {}).get("user_email") or ""
    credits = int((state or {}).get("user_credits") or 0)
    if CREDITS_COST_PER_RUN > 0:
        if not email:
            return [("系统", "请先登录（上方输入邮箱 → 发送登录链接 → 点链接回来 → 点击“从登录状态加载”）。")], gr.update(), state, gr.update(value=""), gr.update(value=""), gr.update(value="")
        if credits < CREDITS_COST_PER_RUN:
            return [("系统", f"次数不足（剩余 {credits}）。请先购买或领取试用。")], gr.update(), state, gr.update(value=""), gr.update(value=""), gr.update(value="")
    try:
        repo_data = analyze_repo(url)
        steps_text = generate_guide(repo_data, env)
        steps = [s.strip() for s in split_steps(steps_text) if s.strip()]

        history = [("系统", "分析完成。以下是初始部署指导（右侧清单可标记完成）：")]
        history.append(("系统", "\n\n".join(steps) if steps else "生成步骤失败，请检查 URL 或仓库内容。"))

        def to_title(s: str) -> str:
            head = s.splitlines()[0].strip()
            head = re.sub(r"^\d+\.\s*", "", head)
            return (head[:80] + ("…" if len(head) > 80 else ""))

        titles = [f"{i+1}. {to_title(s)}" for i, s in enumerate(steps)]
        raw_codes = extract_code_snippets(steps_text)
        codes, warns = sanitize_commands(raw_codes)

        # 扣次
        if CREDITS_COST_PER_RUN > 0:
            add_credits(email, -CREDITS_COST_PER_RUN)
            new_u = get_user(email)
            state["user_credits"] = int(new_u["credits"]) if new_u else max(0, credits - CREDITS_COST_PER_RUN)

        state.update({
            "repo_data": repo_data, "history": history, "user_env": env,
            "steps": steps, "titles": titles, "codes": codes, "warnings": warns,
        })

        lang = "powershell" if env == "Windows" else "bash"
        codes_md = "\n\n".join([f"```{lang}\n{c}\n```" for c in codes]) if codes else "_未提取到命令_"
        warn_md  = "\n".join([f"- {w}" for w in warns]) if warns else "暂无高危命令。"

        return history, gr.update(choices=titles, value=[]), state, gr.update(value=codes_md), gr.update(value=""), gr.update(value=warn_md)
    except Exception as e:
        return [("系统", f"错误: {e}")], gr.update(choices=[], value=[]), state or init_state(), gr.update(value=""), gr.update(value=""), gr.update(value="")

def on_select_step(selected_titles, state):
    try:
        titles = state.get("titles", []) or []
        steps  = state.get("steps", []) or []
        if not selected_titles: return ""
        mapping = {titles[i]: steps[i] for i in range(min(len(titles), len(steps)))}
        return mapping.get(selected_titles[-1], "")
    except Exception:
        return ""

def handle_feedback(history, feedback, state):
    h = list(history or [])
    if not state.get("repo_data"):
        return h + [("系统", "请先输入 URL 并开始分析。")], state, gr.update(value="")
    if feedback and feedback.strip():
        suggestion = troubleshoot(feedback, state.get("repo_data", {})) or "收到。请粘贴完整报错堆栈以便进一步定位。"
        new_h = h + [("用户", feedback), ("系统", suggestion)]
    else:
        new_h = h + [("系统", "请输入具体错误信息。")]
    state["history"] = new_h
    return new_h, state, gr.update(value="")

def guided_diagnose(issue, history, state):
    if not issue: return history
    mapping = {
        "依赖安装失败": "依赖安装失败（pip/conda），报错如版本冲突/找不到轮子/权限不足）。",
        "CUDA/显卡相关": "CUDA/显卡相关问题（nvidia-smi 不可用、CUDA 版本不匹配）。",
        "端口被占用": "端口被占用（地址已被使用，OSError: [Errno 98]/[WinError 10048]）。",
        "环境变量问题": "环境变量缺失或路径错误（.env 未加载、API Key 未设置）。",
        "权限/路径问题": "权限/路径问题（权限拒绝、中文/空格路径、只读目录）。",
        "其他": "其他未分类问题。",
    }
    return (history or []) + [("系统", troubleshoot(mapping.get(issue, issue), state.get("repo_data", {})))]

# ============ 登录态 / 试用 / 支付按钮 ============
def request_magic_link(email):
    if not email or "@" not in email:
        return gr.update(value="请输入有效邮箱。")
    upsert_user(email.strip())
    token = create_magic_token(email.strip())
    link = f"{APP_BASE_URL}/auth/verify?token={token}"
    # 现在先直接返回链接（等接入邮件服务再真正发邮件）
    return gr.update(value=f"请点击登录完成验证：\n\n[{link}]({link})")

def load_session(state):
    try:
        r = _rq.get(f"{APP_BASE_URL}/auth/whoami", timeout=10)
        data = r.json()
        if not data.get("ok"):
            return state, gr.update(value="**账户：未登录 | 剩余次数：0**")
        email = data["email"]; credits = int(data["credits"])
        state["user_email"] = email; state["user_credits"] = credits
        return state, gr.update(value=f"**账户：{email} | 剩余次数：{credits}**")
    except Exception as e:
        return state, gr.update(value=f"读取登录状态失败：{e}")

def claim_trial(state):
    email = state.get("user_email") or ""
    if not email:
        return gr.update(value="请先完成登录→点【从登录状态加载】。"), state
    u = get_user(email)
    if u and int(u.get("trial_used", 0)) == 0:
        add_credits(email, TRIAL_CREDITS)
        set_trial_used(email)
        state["user_credits"] = int(get_user(email)["credits"])
        return gr.update(value=f"🎁 试用已到账 +{TRIAL_CREDITS} 次。当前余额：{state['user_credits']}"), state
    return gr.update(value="试用已领取过，无法重复领取。"), state

def make_pay_link(plan, state):
    email = state.get("user_email") or ""
    if not email:
        return gr.update(value="请先完成登录→点【从登录状态加载】。")
    info = PLAN_CREDIT_MAP.get(plan)
    if not info or not info["price_id"]:
        return gr.update(value=f"未配置 {plan} 的 priceId，请在环境变量中设置。")
    url = f"{APP_BASE_URL}/paddle/checkout/{plan.lower()}?email={quote_plus(email)}"
    return gr.update(value=f"➡️ **点击跳转支付：** [{plan} 购买链接]({url})")

# =========================
# Gradio UI
# =========================
with gr.Blocks(title="Repo AI 助手", theme=theme, css=CSS) as demo:
    gr.Markdown("""
    <div id='header'>
      <h1>GitHub Repo 部署助手</h1>
      <p>输入仓库链接，自动生成一键部署清单与逐步指南。<br>支持邮箱登录、试用与 Paddle 充值。</p>
    </div>
    """)

    with gr.Row():
        login_email = gr.Textbox(label="登录邮箱", placeholder="例如：you@example.com", scale=6)
        send_link_btn = gr.Button("发送登录链接（Magic Link）", variant="primary", scale=3)
        load_session_btn = gr.Button("从登录状态加载", scale=2)
    acct_badge = gr.Markdown("**账户：未登录 | 剩余次数：0**")
    trial_btn = gr.Button(f"领取试用 +{TRIAL_CREDITS} 次")
    pay_msg = gr.Markdown("")

    with gr.Row(equal_height=True):
        url_input = gr.Textbox(label="GitHub URL", placeholder="例如：https://github.com/huggingface/transformers", scale=7)
        env_dropdown = gr.Dropdown(choices=["Windows", "Mac", "Linux"], label="你的系统", value="Windows", scale=2)
        start_btn = gr.Button("开始分析", variant="primary", scale=1)

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="交互指导", height=520)

            with gr.Accordion("错误诊断向导（可选）", open=False):
                issue_select = gr.Dropdown(
                    label="选择一个常见问题类型",
                    choices=["依赖安装失败", "CUDA/显卡相关", "端口被占用", "环境变量问题", "权限/路径问题", "其他"]
                )
                diagnose_btn = gr.Button("一键生成排查步骤")

            with gr.Row():
                feedback_input = gr.Textbox(label="反馈问题（如错误消息）", placeholder="例如：No module named torch", scale=6)
                submit_btn = gr.Button("提交反馈", variant="primary", scale=2)
                reset_btn  = gr.Button("重置/取消", scale=2)

            success_btn = gr.Button("确认成功！")

            with gr.Accordion("可复制命令", open=False):
                cmds_md = gr.Markdown(value="", show_copy_button=True)

            with gr.Row():
                export_md_btn = gr.Button("导出为 Markdown")
                export_sh_btn = gr.Button("导出脚本（自动匹配 .ps1/.sh）")
                export_json_btn = gr.Button("导出 JSON")
            export_md_file = gr.File(label="下载 Markdown", visible=False)
            export_sh_file = gr.File(label="下载脚本", visible=False)
            export_json_file = gr.File(label="下载 JSON", visible=False)

            with gr.Accordion("购买套餐", open=False):
                with gr.Row():
                    buy_basic = gr.Button("购买 BASIC（200 次）")
                    buy_pro   = gr.Button("购买 PRO（1000 次）")

        with gr.Column(scale=5, elem_id="right_pane"):
            checklist = gr.CheckboxGroup(label="部署步骤清单", choices=[], interactive=True, value=[], elem_id="checklist")
            with gr.Accordion("步骤详情", open=True):
                step_detail_md = gr.Markdown(value="", elem_id="step_detail", show_copy_button=True)
            with gr.Accordion("安全提示", open=False):
                warnings_md = gr.Markdown(value="")
            with gr.Accordion("使用小贴士", open=False):
                gr.Markdown("""
- 建议使用虚拟环境（venv/conda）。
- 首次运行失败，优先检查 Python 版本与 pip 源。
- Windows 建议使用 PowerShell（管理员）安装依赖。
- CUDA/CuDNN 相关，请核对驱动与 `torch` 映射版本。
                """)

    state_component = gr.State(init_state())

    # 事件绑定
    send_link_btn.click(request_magic_link, inputs=[login_email], outputs=[pay_msg])
    load_session_btn.click(load_session, inputs=[state_component], outputs=[state_component, acct_badge])

    trial_btn.click(claim_trial, inputs=[state_component], outputs=[pay_msg, state_component])

    buy_basic.click(lambda s: make_pay_link("BASIC", s), inputs=[state_component], outputs=pay_msg)
    buy_pro.click(lambda s: make_pay_link("PRO", s), inputs=[state_component], outputs=pay_msg)

    start_btn.click(
        start_analysis,
        inputs=[url_input, env_dropdown, state_component],
        outputs=[chatbot, checklist, state_component, cmds_md, step_detail_md, warnings_md],
        concurrency_limit=10,
    )
    checklist.select(on_select_step, inputs=[checklist, state_component], outputs=step_detail_md)
    submit_btn.click(handle_feedback, inputs=[chatbot, feedback_input, state_component],
                     outputs=[chatbot, state_component, feedback_input])
    diagnose_btn.click(guided_diagnose, inputs=[issue_select, chatbot, state_component], outputs=[chatbot])

    success_btn.click(lambda h, s: h + [("系统", "恭喜！部署成功。")], inputs=[chatbot, state_component], outputs=chatbot)

    export_md_btn.click(export_markdown, inputs=[state_component], outputs=export_md_file)
    export_sh_btn.click(export_script,   inputs=[state_component], outputs=export_sh_file)
    export_json_btn.click(export_json,   inputs=[state_component], outputs=export_json_file)

    reset_btn.click(
        lambda: ([], gr.update(choices=[], value=[]), init_state(), "", "Windows", "", "",
                 gr.update(value=None, visible=False),
                 gr.update(value=None, visible=False),
                 gr.update(value=None, visible=False),
                 "**账户：未登录 | 剩余次数：0**",""),
        outputs=[chatbot, checklist, state_component, url_input, env_dropdown,
                 cmds_md, step_detail_md, export_md_file, export_sh_file, export_json_file,
                 acct_badge, pay_msg]
    )

    # 页面加载时，尝试自动读取登录态（可选）
    def _auto_load(state):
        try:
            r = _rq.get(f"{APP_BASE_URL}/auth/whoami", timeout=6)
            data = r.json()
            if data.get("ok"):
                state["user_email"] = data["email"]
                state["user_credits"] = int(data["credits"])
                return state, f"**账户：{data['email']} | 剩余次数：{data['credits']}**"
        except Exception:
            pass
        return state, "**账户：未登录 | 剩余次数：0**"
    demo.load(_auto_load, inputs=[state_component], outputs=[state_component, acct_badge])

# =========================
# FastAPI（Auth / Paddle / 挂载）
# =========================
init_db()
api = FastAPI()

# 允许跨域（便于以后前门反代）
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# 根路径跳转到 /ui（Render 根路径“加载中”问题）
@api.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui")

# ---- Auth ----
@api.post("/auth/request")
async def auth_request(data: dict):
    email = (data or {}).get("email", "").strip()
    if "@" not in email:
        return JSONResponse({"ok": False, "error": "invalid_email"}, status_code=400)
    upsert_user(email)
    token = create_magic_token(email)
    verify_url = f"{APP_BASE_URL}/auth/verify?token={token}"
    # 现在直接返回链接（你之后接入邮件服务时改成真正发邮件）
    return {"ok": True, "verify_url": verify_url}

@api.get("/auth/verify")
async def auth_verify(token: str = ""):
    email = use_magic_token(token)
    if not email:
        return HTMLResponse("<h3>链接无效或已过期</h3>", status_code=400)
    resp = RedirectResponse(url="/ui")
    resp.set_cookie(SESSION_COOKIE, _sign(email), httponly=True, max_age=7*24*3600, samesite="lax")
    return resp

@api.get("/auth/whoami")
async def whoami(session: str | None = Cookie(default=None, alias=SESSION_COOKIE)):
    email = _unsign(session)
    if not email:
        return {"ok": False, "email": None, "credits": 0}
    u = get_user(email) or {"credits": 0}
    return {"ok": True, "email": email, "credits": int(u.get("credits", 0))}

# ---- Paddle ----
def _checkout_html(price_id: str, plan: str, email: str) -> str:
    if not PADDLE_CLIENT_TOKEN or not price_id:
        return """<!doctype html><meta charset='utf-8'>
<h3>Paddle 未配置（Sandbox/Live）</h3><p>请在环境变量设置 token 与 priceId 再试。</p>"""
    return f"""<!doctype html><html><head><meta charset="utf-8"/><title>Checkout - {plan}</title>
<script src="https://cdn.paddle.com/paddle/v2/paddle.js"></script></head><body>
<script>
  Paddle.Initialize({{ token: "{PADDLE_CLIENT_TOKEN}" }});
  const email = decodeURIComponent("{quote_plus(email or '')}");
  const openCheckout = () => Paddle.Checkout.open({{
    settings: {{ displayMode: "overlay", variant: "one-page" }},
    items: [{{ priceId: "{price_id}", quantity: 1 }}],
    customer: {{ email }},
    customData: {{ email, plan: "{plan}" }}
  }});
  window.onload = openCheckout;
</script>
<p>若未自动弹出支付窗口，请 <a href="#" onclick="openCheckout();return false;">点此重试</a>。</p>
</body></html>"""

@api.get("/paddle/checkout/basic", response_class=HTMLResponse)
async def paddle_checkout_basic(email: str = ""):
    return HTMLResponse(content=_checkout_html(PADDLE_PRICE_BASIC, "BASIC", email))

@api.get("/paddle/checkout/pro", response_class=HTMLResponse)
async def paddle_checkout_pro(email: str = ""):
    return HTMLResponse(content=_checkout_html(PADDLE_PRICE_PRO, "PRO", email))

@api.post("/paddle/webhook")
async def paddle_webhook(request: Request):
    raw = await request.body()
    sig = request.headers.get("Paddle-Signature")
    if not verify_paddle_signature(sig, raw):
        return JSONResponse({"ok": False, "error": "invalid signature"}, status_code=400)
    payload = json.loads(raw.decode("utf-8"))
    event_id = payload.get("event_id")
    if not mark_event_processed(event_id):
        return JSONResponse({"ok": True, "duplicate": True})
    etype = payload.get("event_type")
    data = payload.get("data", {}) or {}
    if etype == "transaction.completed":
        custom_data = data.get("custom_data") or {}
        email = custom_data.get("email")
        items = data.get("items", []) or []
        total_added = 0
        for it in items:
            price = (it.get("price") or {})
            price_id = price.get("id")
            if price_id in PRICEID_TO_PLAN:
                _, add = PRICEID_TO_PLAN[price_id]
                total_added += int(add) * int(it.get("quantity") or 1)
        if email and total_added > 0:
            upsert_user(email)
            add_credits(email, total_added)
    return JSONResponse({"ok": True})

# ---- Mount Gradio at /ui ----
app = gr.mount_gradio_app(api, demo, path="/ui")

if __name__ == "__main__":
    init_db()
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")

