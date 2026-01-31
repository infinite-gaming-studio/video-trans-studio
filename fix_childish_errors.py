import json

notebook_path = 'Video_Trans_Studio.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 使用完全转义的安全字符串
new_lines = [
    "# @title 🚀 1. 环境初始化\n",
    "import os\n",
    "import sys\n",
    "import numpy as np\n",
    "from packaging import version\n",
    "\n",
    "os.chdir('/content')\n",
    "if not os.path.exists('video-trans-studio'):\n",
    "    get_ipython().system('git clone https://github.com/infinite-gaming-studio/video-trans-studio.git')\n",
    "\n",
    "get_ipython().run_line_magic('cd', 'video-trans-studio')\n",
    "print(\"🔄 正在同步仓库最新代码...\")\n",
    "get_ipython().system('git fetch --all && git reset --hard origin/main && git pull')\n",
    "\n",
    "def check_environment():\n",
    "    try:\n",
    "        import transformers, accelerate\n",
    "        v_trans = version.parse(transformers.__version__)\n",
    "        v_accel = version.parse(accelerate.__version__)\n",
    "        return v_trans >= version.parse(\"4.46.0\") and v_accel >= version.parse(\"0.33.0\")\n",
    "    except:\n",
    "        return False\n",
    "\n",
    "needed_dirs = ['MuseTalk', 'index-tts', 'checkpoints']\n",
    "is_dirs_ready = all(os.path.exists(d) for d in needed_dirs)\n",
    "\n",
    "if not check_environment() or not is_dirs_ready:\n",
    "    print(\"⚠️ 环境检测不通过：正在重构基础环境以支持 Index-TTS2...\")\n",
    "    get_ipython().system('pip uninstall -y transformers tokenizers numpy jax -q')\n",
    "    get_ipython().system('bash setup_colab.sh')\n",
    "    print(\"\\n\" + \"!\"*50)\n",
    "    print(\"✅ 基础环境构建完成！\")\n",
    "    print(\"⚠️ 请点击上方菜单栏：'运行时' -> '重新启动会话' (Runtime -> Restart Session)")\n",
    "    print(\"⚠️ 重启后，再次运行此单元格即可。\")\n",
    "    print(\"!\"*50)\n",
    "else:\n",
    "    import transformers\n",
    "    print(f\"✅ 环境就绪！Transformers: {transformers.__version__}, NumPy: {np.__version__}\")\n"
]

for cell in nb['cells']:
    if cell.get('metadata', {}).get('id') == 'setup':
        cell['source'] = new_lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
print("Successfully fixed syntax errors.")
