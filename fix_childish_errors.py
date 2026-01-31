import json
import os

notebook_path = 'Video_Trans_Studio.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# --- 1. 定义新的同步单元格代码 ---
sync_code = """# @title 🔄 0.1 同步最新代码
import os
target_path = '/content/video-trans-studio'

if not os.path.exists(target_path):
    os.chdir('/content')
    print("📥 正在克隆仓库...")
    get_ipython().system('git clone https://github.com/infinite-gaming-studio/video-trans-studio.git')

os.chdir(target_path)
print("🔄 正在强制同步仓库最新代码...")
get_ipython().system('git fetch --all && git reset --hard origin/main')
"""

# --- 2. 定义环境初始化单元格代码 (移除同步逻辑) ---
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
    print("❌ 错误：未找到项目目录，请先运行上方的同步代码单元格。")

# 2. 高效版本检测
def get_env_status():
    pkg_name = "transformers"
    min_ver = "4.46.0"
    
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
        
        # 检查 LivePortrait 目录
        if not os.path.exists('LivePortrait'):
            return "NEEDS_INSTALL", "缺失 LivePortrait 组件"
            
        return "READY", f"环境就绪 (Transformers {disk_ver})"
    except PackageNotFoundError:
        return "NEEDS_INSTALL", "未检测到核心依赖"

status, detail = get_env_status()

if status == "NEEDS_INSTALL":
    print(f"❌ 环境检测失败: {detail}")
    print("🔄 正在执行深度安装/修复...")
    get_ipython().system('bash setup_colab.sh')
    print("\n✅ 安装脚本执行完毕，请点击 [运行时] -> [重新启动会话] 以激活新版本！")
elif status == "NEEDS_RESTART":
    print(f"⚠️ {detail}")
    print("="*60)
    print("👉 请点击上方工具栏: [运行时] -> [重新启动会话] 👈")
    print("="*60)
else:
    print(f"✅ {detail}")
    print(f"📂 工作目录: {os.getcwd()}")
"""

def string_to_lines(code):
    return [line + "\n" for line in code.split("\n")]

# --- 3. 逻辑：更新或插入单元格 ---
cells = nb['cells']
new_cells = []
sync_cell_found = False
setup_cell_found = False

# 首先检查是否已经有同步单元格 (通过 ID 或特定内容识别)
for cell in cells:
    if cell.get('metadata', {}).get('id') == 'sync-code':
        cell['source'] = string_to_lines(sync_code)
        sync_cell_found = True
    if cell.get('metadata', {}).get('id') == 'setup':
        cell['source'] = string_to_lines(setup_code)
        setup_cell_found = True

# 如果没有同步单元格，在 setup 单元格之前插入
if not sync_cell_found:
    new_sync_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "cellView": "form",
            "id": "sync-code"
        },
        "outputs": [],
        "source": string_to_lines(sync_code)
    }
    # 寻找 setup 单元格的索引
    setup_idx = 0
    for i, cell in enumerate(cells):
        if cell.get('metadata', {}).get('id') == 'setup':
            setup_idx = i
            break
    cells.insert(setup_idx, new_sync_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Successfully split Git sync and Environment setup into two cells.")