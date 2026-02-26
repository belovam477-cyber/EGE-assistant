import json

INPUT_FILE = "data.jsonl"
OUTPUT_FILE = "data_clean.jsonl"

with open(INPUT_FILE, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

valid_count = 0
error_count = 0

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
            f.write(line + '\n')
            valid_count += 1
        except json.JSONDecodeError as e:
            print(f"Строка {i} повреждена: {e}")
            print(f"Содержимое: {repr(line[:200])}...")  # покажем начало
            error_count += 1

print(f"Готово. Корректных строк: {valid_count}, ошибок: {error_count}")