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

# 1. 路径归位
target_path = '/content/video-trans-studio'
if os.path.exists(target_path):
    os.chdir(target_path)
    if target_path not in sys.path:
        sys.path.insert(0, target_path)
else:
    os.chdir('/content')

# 2. 增强版环境检测逻辑
def check_environment():
    results = {"ok": True, "msg": "环境就绪", "diag": []}
    
    # 检查核心目录
    needed_dirs = ['LivePortrait', 'index-tts', 'checkpoints']
    for d in needed_dirs:
        if not os.path.exists(d):
            results["ok"] = False
            results["msg"] = f"缺失组件: {d}"
            return results

    # 检查 Transformers 版本 (重点拷打)
    try:
        t_ver = get_version("transformers")
        results["diag"].append(f"Transformers (Disk): {t_ver}")
        if version.parse(t_ver) < version.parse("4.41.0"):
            results["ok"] = False
            results["msg"] = f"Transformers 版本过低 ({t_ver})，需要至少 4.41.0"
            return results
    except Exception as e:
        results["ok"] = False
        results["msg"] = f"无法读取 Transformers 版本: {e}"
        return results

    # 检查内存缓存
    if 'transformers' in sys.modules:
        import transformers
        m_ver = transformers.__version__
        results["diag"].append(f"Transformers (Memory): {m_ver}")
        if version.parse(m_ver) < version.parse("4.41.0"):
            results["msg"] = "⚠️ 安装已完成，但当前会话仍加载旧版本。请务必点击上方 [运行时] -> [重新启动会话]"
            # 注意：内存过低不代表 ok=False，因为安装已经是对的了，只需要重启
            
    return results

# 运行检测
res = check_environment()
print(f"🔍 诊断信息: {" | ".join(res['diag'])}")

if not res["ok"]:
    print(f"❌ 环境检测不通过: {res['msg']}")
    print("🔄 开始紧急修复环境 (这可能需要几分钟)...")
    
    # 检查仓库
    if not os.path.exists('.git'):
        get_ipython().system('git clone https://github.com/infinite-gaming-studio/video-trans-studio.git .')
    
    get_ipython().system('git fetch --all && git reset --hard origin/main')
    
    # 执行安装，不使用 -q 以便看到报错
    get_ipython().system('bash setup_colab.sh')
    
    print("\n" + "!"*50)
    print("✅ 修复脚本执行完毕！")
    print("⚠️ 请点击上方菜单栏 [运行时] -> [重新启动会话] (Runtime -> Restart Session)")
    print("⚠️ 重启后，再次运行此单元格即可。")
    print("!"*50)
else:
    if "重新启动会话" in res["msg"]:
        print(f"\n{'#'*60}")
        print(f"👉 {res['msg']} 👈")
        print(f"{ '#'*60}\n")
    else:
        print(f"✅ {res['msg']}！可以开始处理视频。")
"""

source_lines = [line + "\n" for line in setup_code.split("\n")]
if source_lines[-1] == "\n":
    source_lines = source_lines[:-1]

for cell in nb['cells']:
    if cell.get('metadata', {}).get('id') == 'setup':
        cell['source'] = source_lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
print("Successfully overhauled setup logic with diagnostic mode.")