import json
import os

notebook_path = 'Video_Trans_Studio.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

setup_code = """# @title 🚀 1. 环境初始化
import os
import sys
from importlib.metadata import version as get_version, PackageNotFoundError
from packaging import version

# 1. 路径自适应
target_path = '/content/video-trans-studio'
if os.path.exists(target_path):
    os.chdir(target_path)
    if target_path not in sys.path:
        sys.path.insert(0, target_path)
else:
    os.chdir('/content')

# 2. 高效版本检测 (不依赖加载模块，不依赖文件扫描)
def get_env_status():
    pkg_name = "transformers"
    min_ver = "4.41.0"
    
    try:
        # 检查磁盘版本
        disk_ver = get_version(pkg_name)
        if version.parse(disk_ver) < version.parse(min_ver):
            return "NEEDS_INSTALL", f"磁盘版本过低: {disk_ver}"
        
        # 检查内存版本 (如果已加载)
        if pkg_name in sys.modules:
            m_ver = getattr(sys.modules[pkg_name], "__version__", None)
            if m_ver and version.parse(m_ver) < version.parse(min_ver):
                return "NEEDS_RESTART", f"安装已就绪 ({disk_ver})，但内存仍加载旧版 ({m_ver})"
        
        # 检查 LivePortrait 目录 (作为最后的完整性检查)
        if not os.path.exists('LivePortrait'):
            return "NEEDS_INSTALL", "缺失 LivePortrait 组件"
            
        return "READY", f"环境就绪 (Transformers {disk_ver})"
    except PackageNotFoundError:
        return "NEEDS_INSTALL", "未检测到核心依赖"

status, detail = get_env_status()

if status == "NEEDS_INSTALL":
    print(f"❌ 环境检测失败: {detail}")
    print("🔄 正在执行深度安装/修复...")
    if not os.path.exists('.git'):
        get_ipython().system('git clone https://github.com/infinite-gaming-studio/video-trans-studio.git .')
    get_ipython().system('git fetch --all && git reset --hard origin/main')
    get_ipython().system('bash setup_colab.sh')
    print("\n✅ 安装脚本执行完毕，请点击 [运行时] -> [重新启动会话] 以激活新版本！")
elif status == "NEEDS_RESTART":
    print(f"⚠️ {detail}")
    print("="*60)
    print("👉 检测到版本冲突！请务必点击上方工具栏: [运行时] -> [重新启动会话] 👈")
    print("="*60)
else:
    print(f"✅ {detail}")
    print(f"📂 当前工作目录: {os.getcwd()}")
"""

source_lines = [line + "\n" for line in setup_code.split("\n")]
if source_lines[-1] == "\n":
    source_lines = source_lines[:-1]

for cell in nb['cells']:
    if cell.get('metadata', {}).get('id') == 'setup':
        cell['source'] = source_lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
print("Successfully implemented high-efficiency metadata detection.")
