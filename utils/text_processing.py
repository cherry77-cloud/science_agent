import re


def extract_prompt(text, word):
    code_block_pattern = rf"```{word}(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    extracted_code = "\n".join(code_blocks).strip()
    return extracted_code


def bytes2human(n_bytes):
    if n_bytes is None:
        return "N/A"
    if n_bytes == 0:
        return "0B"
    iec_units = ["B", "K", "M", "G", "T", "P", "E"]
    i = 0
    while n_bytes >= 1024 and i < len(iec_units) - 1:
        n_bytes /= 1024.0
        i += 1
    if i == 0:
        return f"{n_bytes:.0f}{iec_units[i]}"
    else:
        return f"{n_bytes:.1f}{iec_units[i]}"
