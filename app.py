"""
视频字幕提取 & AI 总结 Web 应用
启动: python app.py
访问: http://localhost:5003
"""

import os
import json
import uuid
import threading
import hashlib
import queue as _queue_mod
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import yt_dlp
from google import genai

# ──────────────────────────────────────────────
# 日志系统
# ──────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

_log_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 文件日志：保留最近 5 个 2MB 文件
_file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'zimu.log'),
    maxBytes=2 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8'
)
_file_handler.setFormatter(_log_formatter)
_file_handler.setLevel(logging.DEBUG)

# 控制台日志
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)
_console_handler.setLevel(logging.INFO)

logger = logging.getLogger('zimu')
logger.setLevel(logging.DEBUG)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)

# Flask/werkzeug 日志也写文件
logging.getLogger('werkzeug').addHandler(_file_handler)


# ──────────────────────────────────────────────
# .env 文件加载（轻量，不覆盖已有环境变量）
# ──────────────────────────────────────────────
def _load_local_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key, value = key.strip(), value.strip()
                # 去除引号包裹
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                if key:
                    os.environ[key] = value
    except Exception as e:
        print(f'[env] 读取 .env 失败: {e}')  # logger not yet initialized


_load_local_env_file()

# ──────────────────────────────────────────────
# Prompt 模板加载（从 prompts.json）
# ──────────────────────────────────────────────
_PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts.json')
_prompts_cache = {}  # 内存缓存
_prompts_mtime = 0   # 文件修改时间

def load_prompts():
    """加载 prompts.json，支持热更新（文件修改后自动重新读取）"""
    global _prompts_cache, _prompts_mtime
    try:
        mtime = os.path.getmtime(_PROMPTS_FILE)
        if mtime != _prompts_mtime:
            with open(_PROMPTS_FILE, 'r', encoding='utf-8') as f:
                _prompts_cache = json.load(f)
            _prompts_mtime = mtime
            try:
                logger.info('已加载 prompts.json（%d 个模板）', len([k for k in _prompts_cache if not k.startswith('_')]))
            except Exception:
                pass
    except Exception as e:
        try:
            logger.error('读取 prompts.json 失败: %s', e)
        except Exception:
            print(f'[prompts] 读取 prompts.json 失败: {e}')
    return _prompts_cache


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
MODEL_ID = "models/gemini-3-flash-preview"


def _load_gemini_api_keys() -> list[str]:
    """支持多 API Key：优先环境变量，其次回退到本地默认。"""
    env_keys = os.environ.get('GEMINI_API_KEYS', '').strip()
    if env_keys:
        keys = [k.strip() for k in env_keys.split(',') if k.strip()]
        if keys:
            return keys

    env_single = os.environ.get('GEMINI_API_KEY', '').strip()
    if env_single:
        return [env_single]

    return []


GEMINI_API_KEYS = _load_gemini_api_keys()
GEMINI_CLIENTS = [genai.Client(api_key=k) for k in GEMINI_API_KEYS]


def generate_content_with_fallback(contents, model=MODEL_ID):
    """按 key 顺序尝试调用；遇到配额/限流/失效/超时自动切换下一个 key。"""
    import time as _time
    if not GEMINI_CLIENTS:
        raise RuntimeError('未配置 Gemini API Key，请在 .env 中设置 GEMINI_API_KEYS')
    last_error = None
    for idx, client in enumerate(GEMINI_CLIENTS):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents
            )
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            # 可自动切换的错误类型
            is_switchable = any(x in msg for x in [
                'quota', 'resource_exhausted', '429', 'rate limit', 'too many requests',  # 配额/限流
                '401', '403', 'permission', 'invalid',  'api_key_invalid', 'unauthorized',  # Key 失效
                'timeout', 'deadline', 'timed out', 'connection',  # 网络问题
                '500', '502', '503', '504', 'internal', 'unavailable',  # 服务端错误
            ])
            if idx < len(GEMINI_CLIENTS) - 1:
                logger.warning(f"[Gemini] key#{idx+1} 失败（{'可切换' if is_switchable else '未知'}），尝试下一个 key。原因: {e}")
                _time.sleep(0.5)  # 短暂延迟避免连续打爆
                continue
            # 最后一个 key 也失败
            logger.error(f"[Gemini] 所有 {len(GEMINI_CLIENTS)} 个 key 均失败。最后错误: {e}")

    raise RuntimeError(f"所有 Gemini Key 调用失败: {last_error}")

app = Flask(__name__, static_folder="static")

# 持久化存储目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PROJECTS_FILE = os.path.join(DATA_DIR, 'projects.json')
TAGS_FILE = os.path.join(DATA_DIR, 'tags.json')
THUMB_CACHE_DIR = os.path.join(DATA_DIR, 'thumb_cache')

# 默认标签
DEFAULT_TAGS = ['政治', '科技', '生活']


def load_tags():
    """加载标签列表"""
    if os.path.exists(TAGS_FILE):
        try:
            with open(TAGS_FILE, 'r', encoding='utf-8') as f:
                tags = json.load(f)
                if isinstance(tags, list) and len(tags) > 0:
                    return tags
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_TAGS[:]


def save_tags(tags):
    """保存标签列表"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TAGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)


def load_projects():
    """从磁盘加载所有项目"""
    if os.path.exists(PROJECTS_FILE):
        try:
            with open(PROJECTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_projects(projects):
    """将所有项目保存到磁盘"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PROJECTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(projects, f, ensure_ascii=False, indent=2)


def save_project(project_id, project_data):
    """保存单个项目"""
    projects = load_projects()
    projects[project_id] = project_data
    save_projects(projects)


def _now_iso():
    import time
    return time.strftime('%Y-%m-%d %H:%M:%S')


def _project_id_from_session_id(session_id: str) -> str:
    if session_id.startswith('session_'):
        return session_id[len('session_'):]
    return ''


def infer_title_from_url(url: str) -> str:
    """在还没拿到 yt-dlp 标题前，基于 URL 生成一个短标题。"""
    try:
        from urllib.parse import urlparse, parse_qs
        u = urlparse(url)
        host = (u.hostname or '').lower()
        path = u.path or ''

        # bilibili: /video/BVxxxx
        if 'bilibili.com' in host:
            parts = [p for p in path.split('/') if p]
            if len(parts) >= 2 and parts[0] == 'video' and parts[1].upper().startswith('BV'):
                return f"B站 {parts[1]}"
            return 'B站视频'

        # b23 short link
        if host.endswith('b23.tv'):
            code = path.strip('/').split('/')[0] if path.strip('/') else ''
            return f"B站短链 {code}" if code else 'B站短链'

        # youtube
        if 'youtube.com' in host:
            qs = parse_qs(u.query or '')
            vid = (qs.get('v') or [''])[0]
            return f"YouTube {vid}" if vid else 'YouTube 视频'
        if host.endswith('youtu.be'):
            vid = path.strip('/').split('/')[0] if path.strip('/') else ''
            return f"YouTube {vid}" if vid else 'YouTube 视频'
    except Exception:
        pass
    return '未命名项目'


def format_upload_date(date_str: str) -> str:
    """将 yt-dlp 的 upload_date (YYYYMMDD) 转为 YYYY-MM-DD。"""
    if not date_str or not isinstance(date_str, str):
        return ''
    s = date_str.strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s


def extract_video_meta(info: dict) -> dict:
    """从 yt-dlp info 中提取前端需要的元数据。"""
    if not isinstance(info, dict):
        return {'title': '', 'uploader': '', 'upload_date': '', 'thumbnail': '',
                'uploader_url': '', 'uploader_avatar': ''}

    title = info.get('title') or ''
    uploader = info.get('uploader') or info.get('channel') or info.get('uploader_id') or ''
    upload_date = format_upload_date(info.get('upload_date') or '')

    # 作者主页链接
    uploader_url = info.get('uploader_url') or info.get('channel_url') or ''

    # B站：通过 uploader_id (mid) 构造 space 链接
    if not uploader_url:
        webpage_url = info.get('webpage_url') or info.get('original_url') or ''
        mid = info.get('uploader_id') or ''
        if 'bilibili.com' in webpage_url and mid:
            uploader_url = f'https://space.bilibili.com/{mid}'

    # 作者头像
    uploader_avatar = ''
    # B站 API 会在 info 中返回 uploader 头像
    for key in ('uploader_thumbnail', 'channel_thumbnail', 'avatar'):
        if info.get(key):
            uploader_avatar = info[key]
            break
    # 有些 yt-dlp 版本把头像放 thumbnails 列表中带 id='avatar'
    if not uploader_avatar:
        thumbs_list = info.get('thumbnails') or []
        for t in thumbs_list:
            if isinstance(t, dict) and t.get('id') == 'avatar':
                uploader_avatar = t.get('url', '')
                break

    thumb = info.get('thumbnail') or ''
    if not thumb:
        thumbs = info.get('thumbnails')
        if isinstance(thumbs, list) and thumbs:
            cand = thumbs[-1]
            if isinstance(cand, dict):
                thumb = cand.get('url') or ''

    return {
        'title': title,
        'uploader': uploader,
        'upload_date': upload_date,
        'thumbnail': thumb,
        'uploader_url': uploader_url,
        'uploader_avatar': uploader_avatar,
    }


def _thumbnail_cache_path(url: str, ext: str) -> str:
    digest = hashlib.sha1(url.encode('utf-8')).hexdigest()
    return os.path.join(THUMB_CACHE_DIR, f"{digest}.{ext}")


def _download_thumbnail(url: str, referer: str = ''):
    """下载封面并返回 (bytes, content_type)。"""
    import urllib.request

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
    }
    if referer:
        headers['Referer'] = referer

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()
        ctype = resp.headers.get('Content-Type', 'image/jpeg').split(';')[0].strip().lower()
    return data, ctype


def _ext_from_content_type(content_type: str) -> str:
    if content_type == 'image/png':
        return 'png'
    if content_type == 'image/webp':
        return 'webp'
    if content_type == 'image/gif':
        return 'gif'
    return 'jpg'


# 存储每个任务的状态
tasks = {}
# 存储每个会话的聊天历史（key: session_id）
chat_sessions = {}
# Whisper 转录锁（MLX Metal 不支持并发转录，否则会崩溃）
transcribe_lock = threading.Lock()

# ── 任务队列（替代裸线程，防止并发崩溃） ──
task_queue = _queue_mod.Queue()


def _worker_loop():
    """从队列中依次取任务执行，保证不会同时跑多个任务。"""
    while True:
        task_id, url = task_queue.get()
        try:
            process_video_task(task_id, url)
        except Exception as e:
            t = tasks.get(task_id)
            if isinstance(t, dict):
                t['status'] = 'error'
                t['message'] = f'❌ 后台任务异常: {e}'
            logger.exception(f'[worker] 任务 {task_id} 异常: {e}')
        finally:
            task_queue.task_done()


# 启动 1 个 worker（串行执行，避免 MLX Metal 并发崩溃）
threading.Thread(target=_worker_loop, daemon=True).start()


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _extract_first_url(text: str) -> str:
    """从用户输入中提取第一个 http(s) URL（兼容「标题+链接」混合文本）。"""
    import re
    text = text.strip()
    if not text:
        return ''

    # 带协议的完整 URL
    m = re.search(r'https?://[^\s<>"\'\u3000]+', text, re.IGNORECASE)
    if m:
        return m.group(0).rstrip('.,;!?')

    # 不带协议的常见域名
    m = re.search(r'(?:b23\.tv|bilibili\.com|youtube\.com|youtu\.be)/[^\s<>"\'\u3000]*', text, re.IGNORECASE)
    if m:
        return 'https://' + m.group(0).rstrip('.,;!?')

    # 如果输入看起来本身就是个短文本且不含空格，当作 URL 尝试
    if ' ' not in text and '.' in text:
        return text

    return ''


def normalize_url(url):
    """规范化视频 URL，确保有正确的协议和域名前缀"""
    import re
    url = url.strip()

    # 去掉协议头，统一处理
    bare = re.sub(r'^https?://', '', url)

    # B站链接: 确保有 www. 前缀（bilibili.com 不带 www 会 403）
    if bare.startswith('bilibili.com'):
        bare = 'www.' + bare
    elif bare.startswith('m.bilibili.com'):
        # 移动端链接转桌面端
        bare = 'www.' + bare[2:]

    # 确保有 https:// 前缀
    if not bare.startswith(('www.bilibili.com', 'www.youtube.com', 'youtu.be', 'm.youtube.com')):
        # 其他链接，保持原样加 https
        if not url.startswith('http'):
            url = 'https://' + bare
        return url

    return 'https://' + bare


def format_timestamp(seconds):
    """将秒数转换为 MM:SS 格式"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def extract_official_subtitles(url):
    """
    尝试提取视频的官方字幕（自动/手动字幕）。
    返回 (segments_list, source_type) 或 (None, None)
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['zh-Hans', 'zh-CN', 'zh', 'zh-TW', 'en'],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        meta = extract_video_meta(info)
        title = meta.get('title') or '未知标题'

        # 优先用手动字幕，其次自动字幕
        subs = info.get('subtitles', {})
        auto_subs = info.get('automatic_captions', {})

        # 按优先级查找字幕
        lang_priority = ['zh-Hans', 'zh-CN', 'zh', 'zh-TW', 'en']
        chosen_subs = None
        source_type = None
        chosen_lang = None

        for lang in lang_priority:
            if lang in subs:
                chosen_subs = subs[lang]
                source_type = "official"
                chosen_lang = lang
                break

        if chosen_subs is None:
            for lang in lang_priority:
                if lang in auto_subs:
                    chosen_subs = auto_subs[lang]
                    source_type = "auto"
                    chosen_lang = lang
                    break

        if chosen_subs is None:
            return None, None, meta

        # 选择 json3 或 srv1 格式以获取时间戳
        sub_url = None
        for fmt in chosen_subs:
            if fmt.get('ext') == 'json3':
                sub_url = fmt['url']
                break
        if sub_url is None:
            for fmt in chosen_subs:
                if fmt.get('ext') in ('srv1', 'vtt', 'srv2'):
                    sub_url = fmt['url']
                    break

        if sub_url is None:
            return None, None, meta

        # 下载并解析字幕
        import urllib.request
        req = urllib.request.Request(sub_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode('utf-8')

        # 尝试 json3 格式解析
        segments = []
        try:
            data = json.loads(raw)
            events = data.get('events', [])
            for ev in events:
                segs = ev.get('segs', [])
                text = ''.join(s.get('utf8', '') for s in segs).strip()
                if not text or text == '\n':
                    continue
                start_ms = ev.get('tStartMs', 0)
                dur_ms = ev.get('dDurationMs', 0)
                segments.append({
                    'start': start_ms / 1000.0,
                    'end': (start_ms + dur_ms) / 1000.0,
                    'text': text
                })
        except (json.JSONDecodeError, KeyError):
            # 尝试用 VTT/SRT 格式解析
            segments = parse_vtt_subtitles(raw)

        if segments:
            return segments, source_type, meta

        return None, None, meta

    except Exception as e:
        logger.error(f"[字幕提取] 错误: {e}")
        return None, None, {'title': '未知标题', 'uploader': '', 'upload_date': '', 'thumbnail': ''}


def parse_vtt_subtitles(raw_text):
    """简单解析 VTT 格式字幕"""
    import re
    segments = []
    # 匹配时间行: 00:00:01.000 --> 00:00:04.000
    pattern = re.compile(
        r'(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})'
    )
    lines = raw_text.split('\n')
    i = 0
    while i < len(lines):
        match = pattern.search(lines[i])
        if match:
            h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
            start = int(h1)*3600 + int(m1)*60 + int(s1) + int(ms1)/1000
            end = int(h2)*3600 + int(m2)*60 + int(s2) + int(ms2)/1000
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text = ' '.join(text_lines)
            # 去除 VTT 标签
            text = re.sub(r'<[^>]+>', '', text)
            if text:
                segments.append({'start': start, 'end': end, 'text': text})
        i += 1
    return segments


def download_and_transcribe(url, audio_path, task=None):
    """下载音频并用 Whisper 转录"""
    import mlx_whisper
    import tqdm as tqdm_module
    import glob

    # audio_path 应该是不带扩展名的 stem，如 "temp_xxx"
    # 去掉可能存在的扩展名
    audio_stem = os.path.splitext(audio_path)[0]

    # 下载
    if task:
        task['status'] = 'downloading'
        task['message'] = '⬇️ 正在下载音频...'

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_stem + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # yt-dlp 后处理器转码后，实际文件名可能是 stem.m4a
    # 按优先级查找实际生成的音频文件
    actual_audio = None
    for ext in ['m4a', 'mp3', 'wav', 'ogg', 'opus', 'webm', 'mp4']:
        candidate = f"{audio_stem}.{ext}"
        if os.path.exists(candidate):
            actual_audio = candidate
            break
    # 兜底：用 glob 匹配
    if not actual_audio:
        candidates = glob.glob(f"{audio_stem}.*")
        if candidates:
            actual_audio = candidates[0]

    if not actual_audio or not os.path.exists(actual_audio):
        raise FileNotFoundError(f"音频下载失败：未找到文件 {audio_stem}.*")

    logger.info(f"[转录] 音频文件: {actual_audio} ({os.path.getsize(actual_audio)} bytes)")

    if task:
        task['status'] = 'transcribing'
        task['message'] = '🧠 正在使用 Whisper 进行语音转录...'
        task['progress'] = '模型加载中...'

    # ── Monkey-patch tqdm 以捕获转录进度 ──
    _original_tqdm = tqdm_module.tqdm

    class _ProgressTqdm(_original_tqdm):
        def update(self, n=1):
            super().update(n)
            if task and self.total and self.total > 0:
                pct = min(self.n / self.total * 100, 100)
                task['transcribe_percent'] = round(pct, 1)
                task['transcribe_current'] = self.n
                task['transcribe_total'] = self.total
                task['progress'] = f'转录中 {pct:.0f}%'

    tqdm_module.tqdm = _ProgressTqdm
    try:
        # 转录（verbose=False 启用 tqdm）
        result = mlx_whisper.transcribe(
            actual_audio,
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            verbose=False,
            language="zh"
        )
    finally:
        tqdm_module.tqdm = _original_tqdm

    segments = result.get('segments', [])

    # 更新已转录的 segments 到 task
    if task:
        task['segments'] = [{'start': s['start'], 'end': s.get('end', 0), 'text': s['text'].strip()} for s in segments]
        task['progress'] = f'转录完成，共 {len(segments)} 段'
        task['transcribe_percent'] = 100

    # 清理
    if os.path.exists(actual_audio):
        os.remove(actual_audio)

    return segments


def build_timestamped_transcript(segments):
    """将 segments 转换为带时间戳的文本"""
    lines = []
    for seg in segments:
        ts = format_timestamp(seg['start'])
        lines.append(f"[{ts}] {seg['text'].strip()}")
    return "\n".join(lines)


def get_summary_prompt(timestamped_transcript, title='', uploader='', upload_date=''):
    """生成总结 prompt（从 prompts.json 读取模板）"""
    prompts = load_prompts()
    template = prompts.get('summary_prompt', '')
    if not template:
        logger.warning('prompts.json 中未找到 summary_prompt，使用内置默认')
        template = '请总结以下视频字幕内容，使用中文 Markdown 格式：\n\n{transcript}'
    return (template
            .replace('{transcript}', timestamped_transcript)
            .replace('{title}', title or '未知标题')
            .replace('{uploader}', uploader or '未知作者')
            .replace('{upload_date}', upload_date or '未知日期'))


# ──────────────────────────────────────────────
# 后台处理任务
# ──────────────────────────────────────────────

def process_video_task(task_id, url):
    """在后台线程中处理整个流程"""
    task = tasks[task_id]

    try:
        # 步骤 0: 规范化 URL
        url = normalize_url(url)
        task['video_url'] = url
        logger.info(f"[任务 {task_id}] 规范化 URL: {url}")

        # 步骤 1: 检测官方字幕
        task['status'] = 'checking_subtitles'
        task['message'] = '🔍 正在检测视频是否有官方字幕...'
        segments, sub_source, meta = extract_official_subtitles(url)
        if not isinstance(meta, dict):
            meta = {'title': '未知标题', 'uploader': '', 'upload_date': '', 'thumbnail': ''}

        task['title'] = meta.get('title') or '未知标题'
        task['uploader'] = meta.get('uploader', '')
        task['upload_date'] = meta.get('upload_date', '')
        task['thumbnail'] = meta.get('thumbnail', '')
        task['uploader_url'] = meta.get('uploader_url', '')
        task['uploader_avatar'] = meta.get('uploader_avatar', '')

        if segments and len(segments) > 0:
            task['subtitle_source'] = 'official' if sub_source == 'official' else 'auto_generated'
            task['message'] = f'✅ 检测到{"官方" if sub_source == "official" else "自动生成"}字幕，无需转录！'
        else:
            # 步骤 2: 需要下载并转录
            task['subtitle_source'] = 'whisper'
            audio_path = f"temp_{task_id}.m4a"
            # 加锁：MLX Metal GPU 不支持并发转录
            if transcribe_lock.locked():
                task['message'] = '⏳ 等待其他转录任务完成...'
            with transcribe_lock:
                segments = download_and_transcribe(url, audio_path, task=task)
            task['message'] = '✅ 转录完成！'

        # 步骤 3: 构建带时间戳的字幕
        task['status'] = 'summarizing'
        task['message'] = '🤖 正在调用 AI 生成总结...'
        # 保存 segments 原始数据供前端使用
        task['segments'] = [{'start': s['start'], 'end': s.get('end', 0), 'text': s['text'].strip()} for s in segments]
        timestamped_transcript = build_timestamped_transcript(segments)
        task['transcript'] = timestamped_transcript

        # 步骤 4: AI 总结
        prompt = get_summary_prompt(
            timestamped_transcript,
            title=task.get('title', ''),
            uploader=task.get('uploader', ''),
            upload_date=task.get('upload_date', '')
        )
        response = generate_content_with_fallback(prompt, model=MODEL_ID)
        task['summary'] = response.text

        # 步骤 5: AI 自动分类标签
        task['tag'] = ''
        try:
            tags = load_tags()
            if tags:
                prompts = load_prompts()
                classify_template = prompts.get('classify_prompt', '')
                if classify_template:
                    classify_prompt = (classify_template
                        .replace('{title}', task.get('title', ''))
                        .replace('{tags}', '、'.join(tags)))
                    classify_resp = generate_content_with_fallback(classify_prompt, model=MODEL_ID)
                    chosen_tag = classify_resp.text.strip().strip('"').strip("'").strip()
                    # 验证返回的标签在列表中
                    if chosen_tag in tags:
                        task['tag'] = chosen_tag
                        logger.info(f"[任务 {task_id}] AI 分类标签: {chosen_tag}")
                    else:
                        logger.warning(f"[任务 {task_id}] AI 返回标签 '{chosen_tag}' 不在列表中，跳过")
        except Exception as e:
            logger.warning(f"[任务 {task_id}] 自动分类失败: {e}")

        # 完成
        task['status'] = 'done'
        task['message'] = '🎉 处理完成！'

        # 持久化保存项目
        save_project(task_id, {
            'id': task_id,
            'title': task.get('title', '未知标题'),
            'video_url': task.get('video_url', ''),
            'uploader': task.get('uploader', ''),
            'upload_date': task.get('upload_date', ''),
            'thumbnail': task.get('thumbnail', ''),
            'uploader_url': task.get('uploader_url', ''),
            'uploader_avatar': task.get('uploader_avatar', ''),
            'subtitle_source': task.get('subtitle_source', ''),
            'transcript': task.get('transcript', ''),
            'segments': task.get('segments', []),
            'summary': task.get('summary', ''),
            'tag': task.get('tag', ''),
            'created_at': _now_iso(),
            'chat_history': [],
            'status': 'done',
            'message': task.get('message', ''),
            'progress': task.get('progress', ''),
            'transcribe_percent': task.get('transcribe_percent', 0),
        })

    except Exception as e:
        task['status'] = 'error'
        task['message'] = f'❌ 处理失败: {str(e)}'
        logger.error(f"[任务 {task_id}] 错误: {e}", exc_info=True)


# ──────────────────────────────────────────────
# API 路由
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/process', methods=['POST'])
def start_process():
    """启动视频处理任务"""
    data = request.json
    raw = data.get('url', '').strip()

    if not raw:
        return jsonify({'error': '请输入视频链接'}), 400

    # 后端也做一次 URL 提取（防止用户粘贴 "【标题】 https://..." 混合文本）
    url = _extract_first_url(raw)
    if not url:
        return jsonify({'error': '未识别到有效视频链接，请粘贴完整 URL'}), 400

    # 先规范化 URL，避免占位项目里出现不带 www 的 B站链接等
    normalized_url = normalize_url(url)

    # 去重：同一个视频若已有进行中/排队中的任务，直接复用
    running_status = {'queued', 'pending', 'checking_subtitles', 'downloading', 'transcribing', 'summarizing'}
    for exist_id, exist_task in tasks.items():
        if exist_task.get('video_url') == normalized_url and exist_task.get('status') in running_status:
            return jsonify({'task_id': exist_id, 'reused': True})

    task_id = str(uuid.uuid4())[:8]
    created_at = _now_iso()
    tasks[task_id] = {
        'status': 'queued',
        'message': '🧾 已入队，等待处理...',
        'title': '',
        'subtitle_source': '',
        'transcript': '',
        'segments': [],
        'summary': '',
        'video_url': normalized_url,
        'uploader': '',
        'upload_date': '',
        'thumbnail': '',
        'uploader_url': '',
        'uploader_avatar': '',
        'progress': '',
        'transcribe_percent': 0,
        'transcribe_current': 0,
        'transcribe_total': 0,
        'created_at': created_at,
    }

    # 先写入一个“进行中”的项目占位，确保刷新页面也能看到
    # 最终完成后会在 process_video_task 中覆盖为完整数据
    save_project(task_id, {
        'id': task_id,
        'title': infer_title_from_url(normalized_url),
        'video_url': normalized_url,
        'uploader': '',
        'upload_date': '',
        'thumbnail': '',
        'uploader_url': '',
        'uploader_avatar': '',
        'subtitle_source': '',
        'transcript': '',
        'segments': [],
        'summary': '',
        'created_at': created_at,
        'status': 'queued',
        'message': '🧾 已入队，等待处理...',
        'progress': '',
        'transcribe_percent': 0,
        'chat_history': [],
    })

    task_queue.put((task_id, normalized_url))

    return jsonify({'task_id': task_id})


@app.route('/api/status/<task_id>')
def get_status(task_id):
    """查询任务状态"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify(task)


@app.route('/api/projects')
def list_projects():
    """获取所有已保存的项目列表（不含完整内容，只含摘要）"""
    projects = load_projects()

    # 先从磁盘项目生成列表
    by_id = {}
    for pid, p in projects.items():
        by_id[pid] = {
            'id': pid,
            'title': p.get('title', '未知标题') or '未命名项目',
            'subtitle_source': p.get('subtitle_source', ''),
            'created_at': p.get('created_at', ''),
            'video_url': p.get('video_url', ''),
            'thumbnail': p.get('thumbnail', ''),
            'uploader': p.get('uploader', ''),
            'upload_date': p.get('upload_date', ''),
            'uploader_url': p.get('uploader_url', ''),
            'uploader_avatar': p.get('uploader_avatar', ''),
            'status': p.get('status', 'done'),
            'message': p.get('message', ''),
            'progress': p.get('progress', ''),
            'transcribe_percent': p.get('transcribe_percent', 0),
            'tag': p.get('tag', ''),
            'favorite': p.get('favorite', False),
        }

    # 再把内存中的任务状态合并进去（保证刷新页面能看到进行中任务）
    for tid, t in tasks.items():
        row = by_id.get(tid, {
            'id': tid,
            'title': '未命名项目',
            'subtitle_source': '',
            'created_at': t.get('created_at', ''),
            'video_url': t.get('video_url', ''),
            'tag': '',
        })
        # 用最新任务状态覆盖
        row['title'] = t.get('title') or row.get('title') or '未命名项目'
        row['subtitle_source'] = t.get('subtitle_source', row.get('subtitle_source', ''))
        row['created_at'] = t.get('created_at', row.get('created_at', ''))
        row['video_url'] = t.get('video_url', row.get('video_url', ''))
        row['thumbnail'] = t.get('thumbnail', row.get('thumbnail', ''))
        row['uploader'] = t.get('uploader', row.get('uploader', ''))
        row['upload_date'] = t.get('upload_date', row.get('upload_date', ''))
        row['uploader_url'] = t.get('uploader_url', row.get('uploader_url', ''))
        row['uploader_avatar'] = t.get('uploader_avatar', row.get('uploader_avatar', ''))
        row['status'] = t.get('status', row.get('status', ''))
        row['message'] = t.get('message', row.get('message', ''))
        row['progress'] = t.get('progress', row.get('progress', ''))
        row['transcribe_percent'] = t.get('transcribe_percent', row.get('transcribe_percent', 0))
        row['tag'] = t.get('tag', row.get('tag', ''))
        row['favorite'] = row.get('favorite', False)
        by_id[tid] = row

    project_list = list(by_id.values())
    project_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return jsonify(project_list)


@app.route('/api/projects/<project_id>')
def get_project(project_id):
    """获取单个项目的完整数据"""
    projects = load_projects()
    project = projects.get(project_id)
    task = tasks.get(project_id)

    if project is None and task is None:
        return jsonify({'error': '项目不存在'}), 404

    # 以磁盘数据为基础（如果存在），再用内存任务覆盖最新状态
    data = project.copy() if isinstance(project, dict) else {
        'id': project_id,
        'title': '未命名项目',
        'video_url': '',
        'subtitle_source': '',
        'segments': [],
        'transcript': '',
        'summary': '',
        'created_at': '',
        'chat_history': [],
    }

    if task:
        data.update({
            'id': project_id,
            'title': task.get('title') or data.get('title', '未命名项目'),
            'video_url': task.get('video_url') or data.get('video_url', ''),
            'uploader': task.get('uploader', data.get('uploader', '')),
            'upload_date': task.get('upload_date', data.get('upload_date', '')),
            'thumbnail': task.get('thumbnail', data.get('thumbnail', '')),
            'uploader_url': task.get('uploader_url', data.get('uploader_url', '')),
            'uploader_avatar': task.get('uploader_avatar', data.get('uploader_avatar', '')),
            'subtitle_source': task.get('subtitle_source') or data.get('subtitle_source', ''),
            'segments': task.get('segments', data.get('segments', [])),
            'transcript': task.get('transcript', data.get('transcript', '')),
            'summary': task.get('summary', data.get('summary', '')),
            'created_at': task.get('created_at') or data.get('created_at', ''),
            'status': task.get('status', data.get('status', '')),
            'message': task.get('message', data.get('message', '')),
            'progress': task.get('progress', data.get('progress', '')),
            'transcribe_percent': task.get('transcribe_percent', data.get('transcribe_percent', 0)),
            'transcribe_current': task.get('transcribe_current', data.get('transcribe_current', 0)),
            'transcribe_total': task.get('transcribe_total', data.get('transcribe_total', 0)),
            'tag': task.get('tag', data.get('tag', '')),
            'favorite': data.get('favorite', False),
        })
    else:
        data.setdefault('status', data.get('status', 'done'))
        data.setdefault('favorite', data.get('favorite', False))

    return jsonify(data)


@app.route('/api/projects/<project_id>/thumbnail')
def project_thumbnail(project_id):
    """封面代理接口：规避防盗链/CORS，返回项目封面图片。"""
    projects = load_projects()
    project = projects.get(project_id, {}) if isinstance(projects, dict) else {}
    task = tasks.get(project_id, {})

    thumb_url = ''
    video_url = ''
    if isinstance(project, dict):
        thumb_url = project.get('thumbnail', '') or thumb_url
        video_url = project.get('video_url', '') or video_url
    if isinstance(task, dict):
        thumb_url = task.get('thumbnail', '') or thumb_url
        video_url = task.get('video_url', '') or video_url

    if not thumb_url:
        return Response('thumbnail not found', status=404)

    os.makedirs(THUMB_CACHE_DIR, exist_ok=True)
    # 先找缓存
    for ext, ctype in [('jpg', 'image/jpeg'), ('png', 'image/png'), ('webp', 'image/webp'), ('gif', 'image/gif')]:
        cached = _thumbnail_cache_path(thumb_url, ext)
        if os.path.exists(cached):
            with open(cached, 'rb') as f:
                return Response(f.read(), mimetype=ctype, headers={'Cache-Control': 'public, max-age=86400'})

    # 下载并缓存
    try:
        data, ctype = _download_thumbnail(thumb_url, referer=video_url)
        ext = _ext_from_content_type(ctype)
        cache_path = _thumbnail_cache_path(thumb_url, ext)
        with open(cache_path, 'wb') as f:
            f.write(data)
        return Response(data, mimetype=ctype, headers={'Cache-Control': 'public, max-age=86400'})
    except Exception as e:
        logger.warning(f"[封面代理] 下载失败: {e}")
        return Response('thumbnail fetch failed', status=502)


@app.route('/api/projects/<project_id>/avatar')
def project_avatar(project_id):
    """作者头像代理接口（同封面代理逻辑）。"""
    projects = load_projects()
    project = projects.get(project_id, {}) if isinstance(projects, dict) else {}
    task = tasks.get(project_id, {})

    avatar_url = ''
    video_url = ''
    if isinstance(project, dict):
        avatar_url = project.get('uploader_avatar', '') or avatar_url
        video_url = project.get('video_url', '') or video_url
    if isinstance(task, dict):
        avatar_url = task.get('uploader_avatar', '') or avatar_url
        video_url = task.get('video_url', '') or video_url

    if not avatar_url:
        return Response('avatar not found', status=404)

    os.makedirs(THUMB_CACHE_DIR, exist_ok=True)
    for ext, ctype in [('jpg', 'image/jpeg'), ('png', 'image/png'), ('webp', 'image/webp')]:
        cached = _thumbnail_cache_path(avatar_url, ext)
        if os.path.exists(cached):
            with open(cached, 'rb') as f:
                return Response(f.read(), mimetype=ctype, headers={'Cache-Control': 'public, max-age=86400'})

    try:
        data, ctype = _download_thumbnail(avatar_url, referer=video_url)
        ext = _ext_from_content_type(ctype)
        cache_path = _thumbnail_cache_path(avatar_url, ext)
        with open(cache_path, 'wb') as f:
            f.write(data)
        return Response(data, mimetype=ctype, headers={'Cache-Control': 'public, max-age=86400'})
    except Exception as e:
        logger.warning(f"[头像代理] 下载失败: {e}")
        return Response('avatar fetch failed', status=502)


@app.route('/favicon.ico')
def favicon():
    """SVG favicon"""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<rect width="100" height="100" rx="20" fill="#6366f1"/>
<text x="50" y="72" text-anchor="middle" font-size="60" fill="white">📺</text>
</svg>'''
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'public, max-age=604800'})


@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """删除项目"""
    projects = load_projects()
    if project_id in projects:
        del projects[project_id]
        save_projects(projects)
        # 同时清理聊天历史
        session_key = f'session_{project_id}'
        if session_key in chat_sessions:
            del chat_sessions[session_key]
        return jsonify({'ok': True})
    return jsonify({'error': '项目不存在'}), 404


@app.route('/api/projects/<project_id>', methods=['PATCH'])
def update_project(project_id):
    """更新项目属性（标题、标签等）"""
    projects = load_projects()
    project = projects.get(project_id)
    if project is None:
        return jsonify({'error': '项目不存在'}), 404

    data = request.json or {}
    updated = False

    if 'title' in data:
        new_title = str(data['title']).strip()
        if new_title:
            project['title'] = new_title
            updated = True

    if 'tag' in data:
        project['tag'] = str(data['tag']).strip()
        updated = True

    if 'favorite' in data:
        project['favorite'] = bool(data['favorite'])
        updated = True

    if updated:
        save_projects(projects)
    return jsonify({'ok': True, 'title': project.get('title', ''), 'tag': project.get('tag', ''), 'favorite': project.get('favorite', False)})


# ── 标签管理 ──

@app.route('/api/tags')
def get_tags():
    """获取所有标签"""
    return jsonify(load_tags())


@app.route('/api/tags', methods=['POST'])
def add_tag():
    """新增标签"""
    data = request.json or {}
    name = str(data.get('name', '')).strip()
    if not name:
        return jsonify({'error': '标签名不能为空'}), 400
    tags = load_tags()
    if name in tags:
        return jsonify({'error': '标签已存在'}), 400
    tags.append(name)
    save_tags(tags)
    return jsonify({'ok': True, 'tags': tags})


@app.route('/api/tags', methods=['DELETE'])
def delete_tag():
    """删除标签"""
    data = request.json or {}
    name = str(data.get('name', '')).strip()
    if not name:
        return jsonify({'error': '标签名不能为空'}), 400
    tags = load_tags()
    if name not in tags:
        return jsonify({'error': '标签不存在'}), 404
    tags.remove(name)
    save_tags(tags)
    return jsonify({'ok': True, 'tags': tags})


@app.route('/api/classify-all', methods=['POST'])
def classify_all_projects():
    """对所有没有标签的项目进行 AI 分类"""
    projects = load_projects()
    tags = load_tags()
    prompts = load_prompts()
    classify_template = prompts.get('classify_prompt', '')

    if not tags or not classify_template:
        return jsonify({'error': '缺少标签列表或分类 prompt 模板'}), 400

    classified = 0
    failed = 0
    for pid, p in projects.items():
        if p.get('tag') or p.get('status') != 'done':
            continue  # 跳过已有标签或未完成的项目
        title = p.get('title', '')
        if not title or title == '未命名项目':
            continue
        try:
            prompt = (classify_template
                      .replace('{title}', title)
                      .replace('{tags}', '、'.join(tags)))
            resp = generate_content_with_fallback(prompt, model=MODEL_ID)
            chosen = resp.text.strip().strip('"').strip("'").strip()
            if chosen in tags:
                p['tag'] = chosen
                classified += 1
                logger.info(f'[批量分类] {pid} "{title}" → {chosen}')
            else:
                logger.warning(f'[批量分类] {pid} AI 返回 "{chosen}" 不在标签列表中')
                failed += 1
        except Exception as e:
            logger.warning(f'[批量分类] {pid} 分类失败: {e}')
            failed += 1

    save_projects(projects)
    return jsonify({'ok': True, 'classified': classified, 'failed': failed})


@app.route('/api/chat', methods=['POST'])
def chat():
    """多轮对话接口"""
    data = request.json
    session_id = data.get('session_id', '')
    user_message = data.get('message', '').strip()
    transcript = data.get('transcript', '')

    if not user_message:
        return jsonify({'error': '请输入消息'}), 400

    # 初始化或获取聊天历史
    if session_id not in chat_sessions:
        # 如果是已保存项目，加载其历史
        project_id = _project_id_from_session_id(session_id)
        projects = load_projects() if project_id else {}
        persisted = projects.get(project_id) if project_id else None
        persisted_history = (persisted or {}).get('chat_history', [])

        chat_sessions[session_id] = {
            'history': persisted_history[:] if isinstance(persisted_history, list) else [],
            'transcript': transcript
        }
        # 将字幕作为系统上下文（从 prompts.json 读取模板）
        prompts = load_prompts()
        chat_template = prompts.get('chat_system_prompt', '')
        if not chat_template:
            logger.warning('prompts.json 中未找到 chat_system_prompt，使用内置默认')
            chat_template = '你是一位视频内容分析助手。以下是视频字幕：\n\n---\n{transcript}\n---\n\n请基于字幕内容回答用户问题。'
        system_context = chat_template.replace('{transcript}', transcript)

        chat_sessions[session_id]['system_context'] = system_context

    session = chat_sessions[session_id]

    # 构建对话内容
    contents = [session['system_context']]
    for msg in session['history']:
        # 兼容旧格式：可能是 dict 或 str
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'user':
                contents.append(f"用户：{content}")
            elif role == 'assistant':
                contents.append(f"助手：{content}")
            else:
                contents.append(content)
        else:
            contents.append(str(msg))

    contents.append(f"用户：{user_message}")

    try:
        response = generate_content_with_fallback(contents, model=MODEL_ID)
        reply = response.text

        # 保存对话历史
        session['history'].append({'role': 'user', 'content': user_message})
        session['history'].append({'role': 'assistant', 'content': reply})

        # 持久化写回项目（用于刷新后继续追问+回看历史）
        project_id = _project_id_from_session_id(session_id)
        if project_id:
            projects = load_projects()
            project = projects.get(project_id)
            if project is not None:
                project.setdefault('chat_history', [])
                project['chat_history'] = session['history']
                save_projects(projects)

        return jsonify({'reply': reply})

    except Exception as e:
        return jsonify({'error': f'AI 回复失败: {str(e)}'}), 500


# ──────────────────────────────────────────────
# 启动
# ──────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("\n" + "=" * 50)
    print("🚀 视频字幕提取 & AI 总结 Web 应用")
    print("📍 访问地址: http://localhost:5003")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5003, debug=True)
