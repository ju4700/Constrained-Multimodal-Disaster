import os

files_to_clean = [
    r'd:\Development\iiucdatathon\one.py',
    r'd:\Development\iiucdatathon\one_kaggle_notebook.ipynb',
    r'd:\Development\iiucdatathon\three.py',
    r'd:\Development\iiucdatathon\three_kaggle_notebook.ipynb'
]

for file_path in files_to_clean:
    if not os.path.exists(file_path):
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Python script cleanup
    content = content.replace("# \ud83d\udea8 THE DATASET HACK: Prepend category to the context text \ud83d\udea8\n", "")
    content = content.replace("train['context'] = train['category'] + \" - \" + train['context']\n", "")
    content = content.replace("test['context'] = test['category'] + \" - \" + test['context']\n", "")
    
    # Notebook cleanup
    content = content.replace('    "# \\ud83d\\udea8 THE DATASET HACK: Prepend category to the context text \\ud83d\\udea8\\n",\n', "")
    content = content.replace('    "train[\'context\'] = train[\'category\'] + \\" - \\" + train[\'context\']\\n",\n', "")
    content = content.replace('    "test[\'context\'] = test[\'category\'] + \\" - \\" + test[\'context\']\\n",\n', "")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
print("Successfully removed the category hack from all files.")
