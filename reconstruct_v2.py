import json

notebook_path = 'Video_Trans_Studio.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('metadata', {}).get('id') == 'setup':
        cell['source'] = [
            "# @title 🚀 1. 环境初始化\n",
            "import os\n",
            "import sys\n",
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
            "def check_env_integrity():\n",
            "    try:\n",
            "        import numpy\n",
            "        import transformers\n",
            "        from transformers.cache_utils import QuantizedCacheConfig\n",
            "        return True\n",
            "    except:\n",
            "        return False\n",
            "\n",
            "if not check_env_integrity():\n",
            "    print(\"⚠️ 环境不完整或存在冲突，正在执行深度重构 (3-5分钟)...")\n",
            "    # 关键：先卸载所有冲突包\n",
            "    get_ipython().system('pip uninstall -y numpy transformers jax jaxlib tokenizers -q')\n",
            "    # 重新安装最新稳定版\n",
            "    get_ipython().system('pip install numpy>=2.0.0 transformers>=4.46.0 -q')\n",
            "    get_ipython().system('bash setup_colab.sh')\n",
            "    print(\"\n" + \"!\"*50)\n",
            "    print(\"✅ 基础环境构建完成！\")\n",
            "    print(\"⚠️ 重要：请点击上方菜单栏：'运行时' -> '重新启动会话' (Runtime -> Restart Session)")\n",
            "    print(\"⚠️ 重启后再次运行此单元格即可。\")\n",
            "    print(\"!"*50)\n",
            "else:\n",
            "    import transformers\n",
            "    import numpy as np\n",
            "    print(f\"✅ 环境就绪！ Transformers: {transformers.__version__}, NumPy: {np.__version__}\")\n"
        ]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
