#!/bin/bash
# ZimuAI 一键启动脚本
# 用法: ./start.sh

cd "$(dirname "$0")"

# 杀掉旧进程
echo "🔄 清理旧进程..."
lsof -ti:5003 | xargs kill -9 2>/dev/null
sleep 1

# 启动 Flask
echo "🚀 启动 Flask 服务器..."
python app.py &
FLASK_PID=$!
sleep 2

# 检查 Flask 是否启动成功
if ! kill -0 $FLASK_PID 2>/dev/null; then
    echo "❌ Flask 启动失败！"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ ZimuAI 已启动"
echo "📍 本地访问: http://localhost:5003"
echo "📍 局域网:   http://$(ipconfig getifaddr en0 2>/dev/null || echo '?'):5003"
echo "=================================================="
echo ""

# 检查 ngrok 是否可用
if command -v ngrok &>/dev/null; then
    echo "🌍 启动 ngrok 内网穿透..."
    ngrok http 5003 &
    NGROK_PID=$!
    sleep 3

    # 获取公网地址
    PUBLIC_URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print([t['public_url'] for t in d['tunnels'] if t['public_url'].startswith('https')][0])" 2>/dev/null)
    if [ -n "$PUBLIC_URL" ]; then
        echo "=================================================="
        echo "🌐 公网地址: $PUBLIC_URL"
        echo "📋 发给朋友即可使用！"
        echo "=================================================="
    fi
else
    echo "💡 提示: 安装 ngrok 可让朋友远程访问 (brew install ngrok)"
fi

echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待退出，清理子进程
trap "echo ''; echo '🛑 正在停止服务...'; kill $FLASK_PID 2>/dev/null; kill $NGROK_PID 2>/dev/null; exit 0" INT TERM
wait
