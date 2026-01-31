import json

notebook_path = 'Video_Trans_Studio.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

setup_code = """# @title 🚀 1. 环境初始化
import os
import sys
import numpy as np
from packaging import version

os.chdir('/content')
if not os.path.exists('video-trans-studio'):
    get_ipython().system('git clone https://github.com/infinite-gaming-studio/video-trans-studio.git')

get_ipython().run_line_magic('cd', 'video-trans-studio')
print("🔄 正在同步仓库最新代码...")
get_ipython().system('git fetch --all && git reset --hard origin/main && git pull')

def check_environment():
    try:
        import transformers, accelerate
        v_trans = version.parse(transformers.__version__)
        v_accel = version.parse(accelerate.__version__)
        return v_trans >= version.parse("4.46.0") and v_accel >= version.parse("0.33.0")
    except:
        return False

needed_dirs = ['LivePortrait', 'index-tts', 'checkpoints']
is_dirs_ready = all(os.path.exists(d) for d in needed_dirs)

if not check_environment() or not is_dirs_ready:
    print("⚠️ 环境检测不通过：正在重构基础环境以支持 Index-TTS2 & LivePortrait...")
    get_ipython().system('rm -rf MuseTalk')
    get_ipython().system('pip uninstall -y transformers tokenizers numpy jax -q')
    get_ipython().system('bash setup_colab.sh')
    print("\n" + "!"*50)
    print("✅ 基础环境构建完成！")
    print("⚠️ 请点击上方菜单栏：'运行时' -> '重新启动会话' (Runtime -> Restart Session)")
    print("⚠️ 重启后，再次运行此单元格即可。")
    print("!"*50)
else:
    import transformers
    print(f"✅ 环境就绪！Transformers: {transformers.__version__}, NumPy: {np.__version__}")
"""

# Convert string to list of lines as required by nbformat
source_lines = [line + "\n" for line in setup_code.split("\n")]
# Remove the last empty newline added by split if necessary
if source_lines[-1] == "\n":
    source_lines = source_lines[:-1]

for cell in nb['cells']:
    if cell.get('metadata', {}).get('id') == 'setup':
        cell['source'] = source_lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
print("Successfully fixed notebook setup cell using string block.")