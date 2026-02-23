import sys
import os
import yt_dlp
import mlx_whisper
from tqdm import tqdm
from google import genai

# 全局变量用于进度条
pbar = None

def my_hook(d):
    global pbar
    if d['status'] == 'downloading':
        # 获取文件总大小
        total = d.get('total_bytes') or d.get('total_bytes_estimate')
        
        # 如果进度条还没初始化，就初始化一个
        if pbar is None:
            pbar = tqdm(total=total, unit='B', unit_scale=True, desc="⬇️ 下载进度")
        
        # 更新进度条到当前下载量
        downloaded = d.get('downloaded_bytes', 0)
        pbar.n = downloaded
        pbar.refresh()
        
    elif d['status'] == 'finished':
        if pbar:
            pbar.close()
        print("\n✅ 下载完成，正在转换为音频格式...")

def download_audio(url, output_filename="audio.m4a"):
    print(f"\n🎧 正在解析视频地址: {url}")
    
    # 确保之前的进度条被重置
    global pbar
    pbar = None

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_filename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'quiet': True,
        'no_warnings': True,
        'progress_hooks': [my_hook], # 挂载修复后的钩子
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        if pbar: pbar.close()
        raise e
    
    return output_filename

def transcribe_audio(audio_path):
    print("\n" + "="*50)
    print("🧠 M4 神经网络引擎启动 | 模型: large-v3-turbo")
    print("⏳ 正在加载模型并开始转录...")
    print("   (下方将实时显示识别出的文字，请稍候)")
    print("="*50 + "\n")
    
    # verbose=True 会让它实时打印每一句识别结果
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        verbose=True, 
        language="zh"
    )
    
    return result


def format_timestamp(seconds):
    """将秒数转换为 MM:SS 格式"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_timestamped_transcript(segments):
    """将 segments 转换为带时间戳的文本"""
    lines = []
    for seg in segments:
        ts = format_timestamp(seg['start'])
        lines.append(f"[{ts}] {seg['text'].strip()}")
    return "\n".join(lines)


def summarize_with_ai(timestamped_transcript):
    """调用 Gemini API 对字幕进行结构化总结，生成 Markdown"""
    API_KEY = os.environ.get("GEMINI_API_KEY", "")
    MODEL_ID = "models/gemini-3-flash-preview"

    if not API_KEY:
        print("错误：未设置 GEMINI_API_KEY 环境变量")
        return None

    client = genai.Client(api_key=API_KEY)
    
    prompt = f"""你是一位专业的内容分析师。下面是一段视频的字幕文本（带时间戳），是作者的口述内容。

请你以客观第三方的视角，对视频内容进行深入、结构化的总结，生成一份 Markdown 格式的分析文档。

要求：
1. 先给出一个简洁的「视频概述」（2-3句话概括核心主题）
2. 然后提取作者的所有核心观点，每个观点作为一个二级标题（## ），包含：
   - 📌 **观点陈述**：用客观语言重新表述作者的观点
   - 📖 **详细分析**：详细展开作者围绕该观点的论证过程、举的例子、引用的数据或故事
   - 🕐 **对应时间段**：该观点在视频中的大致时间范围（使用时间戳）
   - 💡 **关键论据/依据**：列出作者用来支撑该观点的关键证据、案例或类比
3. 给出一个「总结与启发」部分，概括作者的整体论证逻辑和值得思考的点
4. 请确保对每一个观点都有详细充分的介绍，不要遗漏任何重要论点
5. 在文档最后，增加以下两个独立的部分（使用二级标题）：
   - ## 🧐 AI 理性评价
     以完全客观、理性的语言，对文章的论证逻辑、论据可靠性、推理严谨性进行分析评价。哪些论点有充分依据？哪些存在逻辑跳跃或过度简化？引用的数据和案例是否准确？整体论证结构是否自洽？请直言不讳，不必客气。
   - ## 🎭 内容价值判断：真知灼见还是故弄玄虚？
     坦率评估这篇内容的实际价值。作者是否提供了有深度的洞察，还是在用华丽的比喻和宏大叙事包装简单的道理？对普通观众来说，这些内容是否有实际的指导意义？哪些部分言之有物，哪些部分有"贩卖焦虑"或"故弄玄虚"的嫌疑？请给出你的独立判断。
6. 全文使用中文，Markdown 格式要清晰美观

字幕文本：
---
{timestamped_transcript}
---

请直接输出 Markdown 内容，不要包裹在代码块中。"""

    print("\n" + "="*50)
    print("🤖 正在调用 Gemini AI 进行内容总结...")
    print("   (模型: gemini-3-flash-preview)")
    print("="*50 + "\n")
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"❌ AI 总结失败: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 请提供视频链接")
        print("用法: python zimu.py \"https://www.bilibili.com/video/BV...\"")
        sys.exit(1)

    video_url = sys.argv[1]
    audio_file = "temp_audio.m4a"
    output_txt = "result.txt"
    output_md = "summary.md"
    
    try:
        # 1. 下载 (带进度条)
        if os.path.exists(audio_file):
            os.remove(audio_file)
        download_audio(video_url, audio_file)
        
        # 2. 转录 (带实时字幕和时间戳)
        result = transcribe_audio(audio_file)
        full_text = result['text']
        segments = result.get('segments', [])
        
        # 3. 保存原始字幕文本
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"\n📝 原始字幕已保存至: {output_txt}")
        
        # 4. 构建带时间戳的字幕
        timestamped_transcript = build_timestamped_transcript(segments)
        
        # 5. 调用 AI 进行总结，生成 Markdown
        markdown_content = summarize_with_ai(timestamped_transcript)
        
        if markdown_content:
            with open(output_md, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            print("\n" + "="*50)
            print(f"🎉 大功告成！")
            print(f"   📝 原始字幕: {output_txt}")
            print(f"   📋 AI 总结: {output_md}")
            print(f"\n   👉 直接打开 {output_md} 预览即可了解整个视频内容！")
            print("="*50)
        else:
            print("\n⚠️ AI 总结未成功，但原始字幕已保存。")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        # 清理临时文件
        if os.path.exists(audio_file):
            os.remove(audio_file)