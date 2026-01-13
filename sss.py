import os
import shutil

# Settings
target_folder = r"/home/dima/yyyy/kkkk"
source_folder = r"/home/dima/yyyy/ffff"
allowed_exts = ['.doc', '.docx', '.pdf'] 

files_by_name = {}

# Recursion
for root, dirs, files in os.walk(source_folder):
    for f in files:
        name, ext = os.path.splitext(f)
        ext = ext.lower()

        if ext not in allowed_exts:
            continue
            
        key = name.lower().strip()
        full_path = os.path.join(root, f)
        
        #Save_the_path
        files_by_name.setdefault(key, []).append({'full_path': full_path, 'name': f, 'ext': ext})

print(files_by_name)

# 2. Filteration
for key in list(files_by_name.keys()):
    file_list = files_by_name[key]
    extensions = [item['ext'] for item in file_list]
    
    has_word = any(e in ['.docx', '.doc'] for e in extensions)
    has_pdf = '.pdf' in extensions
    
    if has_word and has_pdf:
        #non-pdf files
        files_by_name[key] = [item for item in file_list if item['ext'] != '.pdf']

# 3. File moving
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

for key, files in files_by_name.items():
    for file_info in files:
        src = file_info['full_path']
        file_name = file_info['name']
        dst = os.path.join(target_folder, file_name)
        
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst) 
            
            shutil.copy2(src, dst)
            print(f"Файл {file_name} перемещен из {os.path.dirname(src)}")




print("Готово!")
