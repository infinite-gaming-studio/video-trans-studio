import json
import os

notebook_path = 'Video_Trans_Studio.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

setup_code = """# @title 🚀 1. 环境初始化
import os
import sys
import numpy as np
from importlib.metadata import version as get_version
from packaging import version

# 1. 路径智能归位
if os.path.exists('/content/video-trans-studio'):
    os.chdir('/content/video-trans-studio')
else:
    os.chdir('/content')

# 2. 检查仓库是否存在
if not os.path.exists('core') and not os.path.exists('video-trans-studio'):
    print("📥 正在初始化仓库...")
    get_ipython().system('git clone https://github.com/infinite-gaming-studio/video-trans-studio.git')
    os.chdir('/content/video-trans-studio')

# 3. 增强版环境检测逻辑 (使用 metadata 避免内存缓存干扰)
def check_environment():
    try:
        # 检测磁盘上安装的版本，而不是内存中的版本
        t_ver = get_version("transformers")
        
        needed_dirs = ['LivePortrait', 'index-tts', 'checkpoints']
        is_dirs_ready = all(os.path.exists(d) for d in needed_dirs)
        
        if not is_dirs_ready:
            return False, "缺少核心模型目录 (LivePortrait/Index-TTS)"
            
        if version.parse(t_ver) < version.parse("4.41.0"):
            return False, f"Transformers 磁盘版本过低: {t_ver}"
            
        # 额外检查：如果内存已经加载了旧版本，提醒重启
        if 'transformers' in sys.modules:
            import transformers
            if version.parse(transformers.__version__) < version.parse("4.41.0"):
                return True, "安装已完成，但检测到旧版本缓存，请务必【重新启动会话】"

        return True, "环境就绪"
    except Exception as e:
        return False, f"检测异常: {e}"

is_ok, reason = check_environment()

if not is_ok:
    print(f"⚠️ 环境需要初始化: {reason}")
    print("🔄 正在同步代码并构建基础环境 (预计 3-5 分钟)...")
    get_ipython().system('git fetch --all && git reset --hard origin/main')
    get_ipython().system('bash setup_colab.sh')
    print("\n" + "!"*50)
    print("✅ 基础环境安装成功！")
    print("⚠️ 关键一步：请点击上方菜单栏 [运行时] -> [重新启动会话] (Runtime -> Restart Session)")
    print("⚠️ 重启后，再次运行此单元格即可看到【环境就绪】。")
    print("!"*50)
elif "重新启动会话" in reason:
    print(f"⚠️ {reason}")
    print("请点击上方工具栏的 [运行时] -> [重新启动会话] ！！")
else:
    print(f"✅ {reason}！")
    import transformers
    print(f"📦 Transformers: {transformers.__version__} | NumPy: {np.__version__}")
"""

source_lines = [line + "\n" for line in setup_code.split("\n")]
if source_lines[-1] == "\n":
    source_lines = source_lines[:-1]

for cell in nb['cells']:
    if cell.get('metadata', {}).get('id') == 'setup':
        cell['source'] = source_lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
print("Successfully fixed notebook setup cell.")
