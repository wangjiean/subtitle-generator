# 📺 ZimuAI — 视频字幕提取 & AI 总结

一站式 Web 工具：输入 B站/YouTube 视频链接 → 自动提取字幕（官方/Whisper 转录） → Gemini AI 深度总结 → 多轮追问聊天。

---

## ✨ 核心功能

| 功能 | 说明 |
|------|------|
| 🎬 字幕提取 | 优先官方/自动字幕，无字幕时本地 Whisper (MLX) 语音转录 |
| 🤖 AI 总结 | Gemini 生成结构化 Markdown，含观点提炼、时间戳、理性评价 |
| 💬 多轮聊天 | 基于字幕上下文追问，历史持久化 |
| 📂 项目管理 | 历史项目列表、回看总结/字幕、继续追问 |
| 🖼 封面 & 头像 | 服务端代理+缓存，规避防盗链 |
| 🔗 作者主页 | 作者名可点击跳转 B站/YouTube 主页，显示作者头像 |
| ⏱ 时间戳跳转 | 总结/聊天中的时间戳可点击跳转到视频对应位置 |
| 📊 转录进度 | 实时进度条显示 Whisper 转录百分比 |
| 🎨 主题切换 | 开心/糖果/夜空三套主题，右上角切换，刷新保留 |
| 📐 三栏布局 | 项目栏/内容栏/聊天栏可拖拽调宽、锁定、隐藏 |
| 🔑 多 Key 轮换 | 支持多个 Gemini API Key，额度满自动切换 |
| 🧾 任务队列 | 串行处理防止 MLX Metal 并发崩溃，队列排队 |
| 🔗 智能 URL | 前后端双重提取，粘贴"标题+链接"混合文本也能识别 |

---

## 📁 项目结构

```
zimu/
├── app.py                 # Flask 后端（核心业务逻辑）
├── static/
│   └── index.html         # 前端单页应用（原生 HTML/CSS/JS）
├── data/
│   ├── projects.json      # 项目持久化存储（历史、聊天记录）
│   └── thumb_cache/       # 封面/头像图片缓存目录
├── logs/
│   └── zimu.log           # 运行日志（自动轮转，最多 5×2MB）
├── .env                   # API Key 配置（不提交 Git）
├── .gitignore             # Git 忽略规则
├── start.sh               # 一键启动脚本（Flask + ngrok）
├── zimu.py                # 早期 CLI 版本（已废弃，保留参考）
├── result.txt             # 早期测试输出
└── summary.md             # 早期测试输出
```

---

## 📄 文件详解

### `app.py` — 后端主文件

| 模块 | 关键函数/变量 | 作用 |
|------|-------------|------|
| **配置** | `_load_local_env_file()` | 启动时读取 `.env`，注入环境变量 |
| | `_load_gemini_api_keys()` | 从环境变量加载多个 Gemini Key |
| | `GEMINI_CLIENTS` | 按 Key 列表创建的 Gemini 客户端池 |
| | `generate_content_with_fallback()` | 多 Key 轮换调用，配额满自动切换 |
| **日志** | `logger` | Python logging，同时写文件 + 控制台 |
| **持久化** | `load_projects()` / `save_project()` | 读写 `data/projects.json` |
| **URL 处理** | `_extract_first_url()` | 从混合文本提取第一个有效 URL |
| | `normalize_url()` | 规范化 URL（B站加 www、移动端转桌面端） |
| **元数据** | `extract_video_meta()` | 从 yt-dlp info 提取标题/作者/封面/头像/主页链接 |
| | `infer_title_from_url()` | URL → 临时短标题（占位用） |
| **字幕** | `extract_official_subtitles()` | 优先抓官方/自动字幕 |
| | `download_and_transcribe()` | 下载音频 + MLX Whisper 转录（带进度） |
| **任务** | `process_video_task()` | 完整处理流水线（字幕→转录→总结→保存） |
| | `task_queue` / `_worker_loop()` | 串行任务队列，防止 Metal 并发崩溃 |
| | `transcribe_lock` | 额外转录互斥锁 |
| **API 路由** | `POST /api/process` | 创建任务（去重、入队） |
| | `GET /api/status/<id>` | 查询任务状态（轮询） |
| | `GET /api/projects` | 项目列表（磁盘+内存合并） |
| | `GET /api/projects/<id>` | 项目详情 |
| | `DELETE /api/projects/<id>` | 删除项目 |
| | `GET /api/projects/<id>/thumbnail` | 封面图代理（缓存） |
| | `GET /api/projects/<id>/avatar` | 作者头像代理（缓存） |
| | `POST /api/chat` | 多轮对话 |
| | `GET /favicon.ico` | SVG favicon |

### `static/index.html` — 前端单页

| 区域 | 说明 |
|------|------|
| CSS 变量 + 主题 | 三套主题（happy/candy/night），`data-theme` 属性切换 |
| 顶栏 | 标题、URL 输入框、主题选择器、布局控制按钮 |
| 左栏（项目列表） | 历史项目列表，进行中任务状态胶囊 |
| 中栏（内容） | 视频卡片（封面/作者头像/主页链接）、Tab 切换（总结/字幕）、进度面板 |
| 右栏（聊天） | 多轮对话输入、Markdown 渲染、时间戳链接化 |
| 布局系统 | 列间拖拽分隔条、列显示/隐藏、锁定宽度、localStorage 持久化 |
| JS 核心函数 | `startProcess()`、`pollStatus()`、`renderSummary()`、`renderVideoMeta()`、`loadProject()` |

### `.env` — 密钥配置

```env
# 多个 key 用逗号分隔
GEMINI_API_KEYS=AIzaXXX,AIzaYYY,AIzaZZZ
```

### `start.sh` — 一键启动

启动 Flask（端口 5003）+ 可选 ngrok 公网隧道。

### `logs/zimu.log` — 运行日志

自动轮转（5 个文件 × 2MB），记录所有请求、任务处理、错误、API 调用。

---

## 🚀 快速启动

### 前置依赖

```bash
pip install flask yt-dlp google-genai mlx-whisper
```

### 配置 API Key

```bash
# 创建 .env 文件
echo 'GEMINI_API_KEYS=你的KEY1,你的KEY2' > .env
```

### 启动

```bash
# 方式 1: 直接运行
python app.py

# 方式 2: 一键脚本（含 ngrok）
chmod +x start.sh
./start.sh
```

访问 http://localhost:5003

---

## 🏗 技术栈

| 层 | 技术 |
|----|------|
| 后端 | Python 3 + Flask |
| 视频抓取 | yt-dlp |
| 语音转录 | MLX Whisper (Apple Silicon 加速) |
| AI 总结/聊天 | Google Gemini API |
| 前端 | 原生 HTML/CSS/JS + marked.js |
| 持久化 | JSON 文件 (`data/projects.json`) |
| 日志 | Python logging + RotatingFileHandler |
| 公网 | ngrok (可选) |

---

## 🔧 架构要点

- **MLX Metal 并发安全**：Whisper 转录走串行队列（`task_queue`），避免 Metal GPU 并发崩溃
- **多 Key 容错**：Gemini API 调用自动轮换 Key，配额满无缝切换
- **封面防盗链**：服务端代理下载 + 磁盘缓存，带 UA/Referer 伪装
- **智能 URL 提取**：前端 `extractFirstUrl()` + 后端 `_extract_first_url()` 双重保障
- **任务去重**：同一 URL 进行中/排队中的任务直接复用
- **布局持久化**：三栏宽度/显隐/锁定状态存 localStorage


## todo:
对视频进行收藏，创建自己的知识库，让自己的大模型可以方便的使用这个数据库获取最新的消息。
