import os
import re
import time
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "volcano").lower()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()

# ========== LLM ==========
if PROVIDER == "volcano":
    client = OpenAI(
        api_key=os.getenv("VOLCANO_API_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3"
    )
else:
    raise ValueError("当前仅支持 volcano 提供商（火山引擎 Ark）")

def call_llm(prompt: str, max_tokens: int):
    try:
        resp = client.chat.completions.create(
            model=os.getenv("VOLCANO_MODEL", "doubao-pro-128k"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"（LLM 调用失败：{e}）"

# ========== GitHub 抓取 ==========
def _gh_get(url, timeout=12, max_retry=3):
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    last_err = None
    for i in range(max_retry):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            # 简单处理限流
            if r.status_code == 403 and "rate limit" in r.text.lower():
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (i + 1))
    raise last_err

def parse_github_url(url):
    parsed = urlparse(url)
    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 2 or parsed.netloc != "github.com":
        raise ValueError("Invalid GitHub URL")
    user, repo = path_parts[0], path_parts[1]
    branch, sub_path = "main", ""
    if len(path_parts) > 2:
        if path_parts[2] == "tree":
            if len(path_parts) > 3:
                branch = path_parts[3]
            if len(path_parts) > 4:
                sub_path = "/".join(path_parts[4:]) + "/"
        else:
            sub_path = "/".join(path_parts[2:]) + "/"
    return user, repo, branch, sub_path

# 读取多个潜在线索文件，提升指南质量
_EXTRA_FILES = [
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
    "Procfile",
    "pyproject.toml",
    "setup.cfg",
    "Pipfile",
    "poetry.lock",
    "environment.yml",
    "package.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "examples/README.md",
    "scripts/README.md",
]

def _try_read_raw(base_raw_url, path, timeout=10):
    try:
        r = requests.get(base_raw_url + path, timeout=timeout)
        if r.status_code == 200:
            return r.text
    except:
        pass
    return ""

def analyze_repo(url):
    user, repo, branch, sub_path = parse_github_url(url)
    api_url = f"https://api.github.com/repos/{user}/{repo}"
    try:
        meta = _gh_get(api_url).json()
    except Exception:
        raise ValueError(f"Repo不存在或无效: https://github.com/{user}/{repo}")

    default_branch = meta.get("default_branch", branch)
    base_raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{default_branch}/{sub_path}"

    readme = _try_read_raw(base_raw_url, "README.md")
    requirements = _try_read_raw(base_raw_url, "requirements.txt") or "No requirements.txt found."

    # 额外线索拼接（截断，避免提示超长）
    extras_blob = []
    for p in _EXTRA_FILES:
        content = _try_read_raw(base_raw_url, p)
        if content:
            # 截断每个文件最多 1200 字符
            extras_blob.append(f"\n==== {p} ====\n{content[:1200]}")
    extras = "\n".join(extras_blob)[:6000]

    return {
        "readme": (readme or "")[:4000],
        "requirements": (requirements or "")[:4000],
        "extras": extras,
        "url": url,
        "default_branch": default_branch,
        "sub_path": sub_path,
    }

# ========== 指南生成 ==========
def generate_guide(repo_data, user_env="Windows", additional_info=""):
    base_prompt = f"""
你是资深“开源项目部署工程师”。基于以下仓库线索，为**新手**编写可落地的部署指南（中文）：
- README（截断）：{repo_data.get('readme','')}
- requirements（截断）：{repo_data.get('requirements','')}
- 其他线索（Dockerfile/pyproject/package.json 等，截断）：{repo_data.get('extras','')}
- 仓库URL：{repo_data.get('url')}

输出要求：
1) **按数字大纲**给出 6–12 个主步骤（1. / 2. ...），每步都包含“命令块 + 解释”。
2) **同时给出 Windows(PowerShell) 与 Mac/Linux(bash)** 的对应命令（分别用代码块标注）。
3) 若涉及深度学习，分别给出 **CPU** 与 **GPU** 安装路径（含 `torch` CUDA 映射提示）。
4) 若线索不足，请给出“通用兜底模板”，保证能从零跑起（clone → venv/conda → 安装 → 运行 → 验证）。
5) 在末尾附上“常见错误排查清单”。

注意安全：提示用户使用虚拟环境；不要执行不明删除/格式化命令。

请直接以**纯文本**输出，主步骤以“数字+空格”开头：
1. ...
2. ...
    """
    if additional_info:
        base_prompt += f"\n（额外上下文：{additional_info}）\n"
    return call_llm(base_prompt, 4000)

def troubleshoot(error_msg, repo_data):
    prompt = f"""
用户报告错误/现象：{error_msg}
仓库：{repo_data.get('url')}
请给出**可操作**的排查步骤（分条），必要时提供 Windows 与 Mac/Linux 的对照命令。
"""
    return call_llm(prompt, 2000)
