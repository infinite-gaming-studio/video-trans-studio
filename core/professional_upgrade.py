
import json
import os

def upgrade_notebook(notebook_path='Video_Trans_Studio.ipynb'):
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        return

    print(f"🔄 Upgrading notebook: {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        # Upgrade Step 1: Environment Initialization
        if cell['cell_type'] == 'code' and '# @title 🚀 1. 环境初始化' in ''.join(cell['source']):
            print("✨ Modernizing 'Environment Initialization' cell...")
            cell['source'] = [
                "# @title 🚀 1. 环境初始化\n",
                "import os\n",
                "import sys\n",
                "from importlib.metadata import version as get_version, PackageNotFoundError\n",
                "from packaging import version\n",
                "\n",
                "# 1. 路径自适应\n",
                "target_path = '/content/video-trans-studio'\n",
                "if os.path.exists(target_path):\n",
                "    os.chdir(target_path)\n",
                "    if target_path not in sys.path:\n",
                "        sys.path.insert(0, target_path)\n",
                "else:\n",
                "    print(\"❌ 错误：未找到项目目录，请先运行上方的同步代码单元格。\")\n",
                "\n",
                "# 2. 高效环境检测\n",
                "def get_env_status():\n",
                "    pkg_name = \"transformers\"\n",
                "    min_ver = \"4.52.0\"\n",
                "    \n",
                "    try:\n",
                "        import subprocess\n",
                "        uv_check = subprocess.run([\"uv\", \"--version\"], capture_output=True)\n",
                "        if uv_check.returncode != 0:\n",
                "             return \"NEEDS_INSTALL\", \"未检测到高性能环境管理工具 uv\"\n",
                "        \n",
                "        disk_ver = get_version(pkg_name)\n",
                "        if version.parse(disk_ver) < version.parse(min_ver):\n",
                "            return \"NEEDS_INSTALL\", f\"核心依赖版本过低: {disk_ver}\"\n",
                "            \n",
                "        return \"READY\", f\"✨ 专业版环境已就绪 (Transformers {disk_ver}, uv enabled)\"\n",
                "    except (PackageNotFoundError, FileNotFoundError):\n",
                "        return \"NEEDS_INSTALL\", \"未检测到核心 AI 基础组件\"\n",
                "\n",
                "status, detail = get_env_status()\n",
                "\n",
                "if status == \"NEEDS_INSTALL\":\n",
                "    print(f\"❌ 环境检测: {detail}\")\n",
                "    print(\"🚀 正在执行高性能环境初始化 (约需 1-2 分钟)...\")\n",
                "    get_ipython().system('bash setup_colab.sh')\n",
                "    print(\"\\n✅ 安装完成！请点击上方 [运行时] -> [重新启动会话] 以激活专业版环境。\")\n",
                "else:\n",
                "    print(f\"✅ {detail}\")\n",
                "    print(f\"📂 工作目录: {os.getcwd()}\")\n"
            ]

        # Upgrade Step 3: Global Run Pipeline
        if cell['cell_type'] == 'code' and '# @title ⚙️ 3. 运行全自动流水线' in ''.join(cell['source']):
            print("✨ Adding 'Emotional Intensity' control to Step 3...")
            cell['source'] = [
                "# @title ⚙️ 3. 运行全自动流水线\n",
                "target_language = \"en\" # @param [\"zh-cn\", \"en\", \"es\", \"fr\", \"ja\"]\n",
                "emo_alpha = 1 # @param {type:\"slider\", min:0, max:1, step:0.1}\n",
                "use_local_translation = True # @param {type:\"boolean\"}\n",
                "\n",
                "import sys\n",
                "import torch\n",
                "import importlib\n",
                "import os\n",
                "\n",
                "# 确保在项目目录中运行\n",
                "if os.getcwd() != '/content/video-trans-studio':\n",
                "    %cd /content/video-trans-studio\n",
                "\n",
                "# 强制重载自定义模块，防止代码缓存\n",
                "modules_to_reload = ['main', 'config', 'core.tts', 'core.lipsync', 'core.utils', 'core.asr', 'core.audio', 'core.translator']\n",
                "for module in modules_to_reload:\n",
                "    if module in sys.modules:\n",
                "        del sys.modules[module]\n",
                "\n",
                "try:\n",
                "    from main import run_pipeline\n",
                "except ImportError as e:\n",
                "    print(f\"❌ 模块加载失败: {e}\")\n",
                "    print(\"\\n🔄 尝试自动紧急修复环境...\")\n",
                "    get_ipython().system('bash setup_colab.sh')\n",
                "    print(\"⚠️ 环境已重置，请务必点击上方 '运行时' -> '重新启动会话'，然后再次运行此单元格。\")\n",
                "    sys.exit()\n",
                "\n",
                "torch.cuda.empty_cache()\n",
                "\n",
                "if 'video_path' in locals() and video_path:\n",
                "    print(f\"🎬 开始处理视频: {video_path}\")\n",
                "    await run_pipeline(video_path, target_lang=target_language, emo_alpha=emo_alpha)\n",
                "else:\n",
                "    print(\"❌ 错误：未定义 video_path，请先成功运行 '第 2 步'。\")\n",
                "\n",
                "print(\"\\n✨ 处理全流程结束！\")"
            ]

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print("✅ Notebook upgrade complete! Please reload the page to see changes.")

if __name__ == '__main__':
    upgrade_notebook()
