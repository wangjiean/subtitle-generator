"""
测试 Gemini API 是否具备联网（Grounding / Google Search）能力
用法: python test_grounding.py
支持多 Key 自动切换
"""

import os, sys, json, time

# ── 加载 .env ──
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

from google import genai
from google.genai import types

# ── 获取所有 API Keys ──
all_keys = []
env_keys = os.environ.get('GEMINI_API_KEYS', '').strip()
if env_keys:
    all_keys = [k.strip() for k in env_keys.split(',') if k.strip()]
if not all_keys:
    single = os.environ.get('GEMINI_API_KEY', '').strip()
    if single:
        all_keys = [single]

if not all_keys:
    print("❌ 未找到 API Key，请在 .env 中设置 GEMINI_API_KEY 或 GEMINI_API_KEYS")
    sys.exit(1)

clients = [(k, genai.Client(api_key=k)) for k in all_keys]
# 只使用第 2、3 个 Key 进行测试
clients = clients[:]
MODEL = "models/gemini-3-flash-preview"


def call_with_fallback(contents, config=None):
    """多 Key 自动切换调用，带超时"""
    from google.genai import types as _types
    last_err = None
    for idx, (key, client) in enumerate(clients):
        try:
            kwargs = dict(model=MODEL, contents=contents)
            if config:
                kwargs['config'] = config
            else:
                kwargs['config'] = _types.GenerateContentConfig(
                    http_options=_types.HttpOptions(timeout=30_000)
                )
            return client.models.generate_content(**kwargs), key
        except Exception as e:
            last_err = e
            tag = f"{key[:8]}...{key[-4:]}"
            print(f"   ⚠️  Key#{idx+1} ({tag}) 失败: {e}")
            if idx < len(clients) - 1:
                print(f"   ➡️  切换到下一个 Key...")
                time.sleep(0.5)
    raise RuntimeError(f"所有 {len(clients)} 个 Key 均失败: {last_err}")


print("=" * 60)
print("🔍 Gemini API 联网能力测试")
print("=" * 60)
print(f"📦 模型: {MODEL}")
print(f"🔑 共 {len(all_keys)} 个 API Key")
for i, k in enumerate(all_keys):
    print(f"   Key#{i+1}: {k[:8]}...{k[-4:]}")
print()

# ── 测试 1: 普通调用（无联网）──
print("━" * 60)
print("【测试 1】普通调用（不启用联网）")
print("━" * 60)
try:
    resp, used_key = call_with_fallback("今天是几号？现在的最新新闻是什么？请简短回答。")
    print(f"✅ 使用 Key: {used_key[:8]}...{used_key[-4:]}")
    print(f"   回复:\n{resp.text}")
except Exception as e:
    print(f"❌ 全部失败: {e}")

print()

# ── 测试 2: 启用 Google Search 联网 ──
print("━" * 60)
print("【测试 2】启用 Google Search（联网搜索）")
print("━" * 60)
try:
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
    resp, used_key = call_with_fallback(
        "今天是几号？今天有什么重大新闻？请简短回答。",
        config=config
    )
    print(f"✅ 使用 Key: {used_key[:8]}...{used_key[-4:]}")
    print(f"   回复:\n{resp.text}")

    # 检查是否有 grounding 元数据
    if resp.candidates and resp.candidates[0].grounding_metadata:
        gm = resp.candidates[0].grounding_metadata
        print(f"\n🌐 Grounding 元数据:")
        if hasattr(gm, 'search_entry_point') and gm.search_entry_point:
            print(f"   搜索入口: 有")
        if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
            print(f"   引用来源: {len(gm.grounding_chunks)} 条")
            for i, chunk in enumerate(gm.grounding_chunks[:5]):
                if hasattr(chunk, 'web') and chunk.web:
                    print(f"   [{i+1}] {chunk.web.title} — {chunk.web.uri}")
        if hasattr(gm, 'web_search_queries') and gm.web_search_queries:
            print(f"   搜索查询: {gm.web_search_queries}")
        print("\n🎉 结论: API 支持联网搜索 ✅")
    else:
        print("\n⚠️  回复成功但未检测到 grounding 元数据，可能未触发搜索")

except Exception as e:
    print(f"❌ 全部失败: {e}")

print()

# ── 测试 3: URL 内容理解能力 ──
print("━" * 60)
print("【测试 3】URL 内容理解能力 (尝试使用 Search)")
print("━" * 60)
try:
    # 必须重新传入带 search 的 config
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
    
    resp, used_key = call_with_fallback(
        "请帮我搜索 https://wttr.in/Beijing?format=3 这个网址的内容，并告诉我上面显示的天气。",
        config=config  # <--- 加上这一行！
    )
    print(f"✅ 使用 Key: {used_key[:8]}...{used_key[-4:]}")
    print(f"   回复:\n{resp.text}")
except Exception as e:
    print(f"❌ 全部失败: {e}")

print()
print("=" * 60)
print("✅ 测试完成")
print("=" * 60)
