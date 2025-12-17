# 用于从输入文本中提取 A/B/C/D 形式的选择题答案，核心逻辑分 4 步，优先级从高到低：
# 文本预处理：去除输入文本的空行，对非空行去首尾空格；若处理后无内容，返回空字符串。

# 首选提取：检查处理后文本的最后一行，若为 A/B/C/D 中任一有效选项，直接返回该选项。
# 次优提取：若最后一行无效，反向检查倒数 5 行，找到首个 A/B/C/D 有效选项并返回。
# 兜底提取：若前两步失败，对倒数 5 行文本用正则匹配 “单独一行、仅含 A/B/C/D（可带前后空格）” 的内容，找到则返回匹配项；均无匹配时返回空字符串。

import re

VALID_CHOICES = {"A", "B", "C", "D"}

def extract_choice_answer(text: str) -> str:
    if not text:
        return ""

    # 1. 拆行并去掉空行
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    # 2. 首选：最后一行
    last_line = lines[-1]
    if last_line in VALID_CHOICES:
        return last_line

    # 3. 次优：倒数 5 行内，严格匹配“单独一行的选项”
    for line in reversed(lines[-5:]):
        if line in VALID_CHOICES:
            return line

    # 4. 兜底（极少触发）：尾部文本中查找独立字母
    tail = "\n".join(lines[-5:])
    m = re.search(r"^\s*([ABCD])\s*$", tail, re.MULTILINE)
    if m:
        return m.group(1)

    return ""
