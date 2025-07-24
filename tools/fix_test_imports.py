#!/usr/bin/env python3
"""
修正測試文件的路徑設置
"""

import os
import re

def fix_test_imports():
    """修正所有測試文件的sys.path設置"""
    
    # 查找所有測試文件
    test_files = []
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                test_files.append(os.path.join(root, file))
    
    # 為每個測試文件添加路徑設置
    for file_path in test_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 檢查是否已有sys.path設置
            if 'sys.path.insert' not in content:
                # 計算相對於項目根目錄的路徑
                depth = file_path.count('/') - 1  # 減去tests/
                parent_path = '../' * depth
                
                # 添加sys.path設置到文件開頭
                lines = content.split('\n')
                
                # 找到第一個import語句的位置
                import_start = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        import_start = i
                        break
                
                # 插入sys.path設置
                sys_path_lines = [
                    '# 設置Python路徑以便導入模組',
                    'import sys',
                    'import os',
                    f'sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "{parent_path}")))',
                    ''
                ]
                
                # 檢查是否已經有import sys
                has_import_sys = any('import sys' in line for line in lines[:import_start+5])
                has_import_os = any('import os' in line for line in lines[:import_start+5])
                
                if has_import_sys:
                    sys_path_lines.remove('import sys')
                if has_import_os:
                    sys_path_lines.remove('import os')
                
                new_lines = lines[:import_start] + sys_path_lines + lines[import_start:]
                new_content = '\n'.join(new_lines)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"✅ 修正: {file_path}")
            else:
                print(f"⏸️  已有路徑設置: {file_path}")
                
        except Exception as e:
            print(f"❌ 錯誤: {file_path} - {e}")

if __name__ == "__main__":
    fix_test_imports()