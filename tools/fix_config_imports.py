#!/usr/bin/env python3
"""
修正config模組的引用
"""

import os
import re

def fix_config_imports():
    """修正config模組引用"""
    
    # 查找所有包含 'import config.config' 的文件
    files_with_config = []
    for root, dirs, files in os.walk('.'):
        if '.git' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'import config.config' in content:
                            files_with_config.append(file_path)
                except:
                    pass
    
    # 修正每個文件
    for file_path in files_with_config:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替換 import config.config 為 import config.config as config
            content = re.sub(r'^import config\.config$', 'import config.config as config', content, flags=re.MULTILINE)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ 修正: {file_path}")
        except Exception as e:
            print(f"❌ 錯誤: {file_path} - {e}")

if __name__ == "__main__":
    fix_config_imports()