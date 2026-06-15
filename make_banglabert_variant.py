with open(r'd:\Development\iiucdatathon\ultimate_notebook.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the XLM-R dict block and remove it (lines 117-141 inclusive, 0-indexed: 116-140)
# We'll detect by scanning for the XLM-R key line and remove the entire dict block
out = []
skip = False
brace_depth = 0

i = 0
while i < len(lines):
    line = lines[i]

    # Detect start of XLM-R dict block
    if 'key         = "xlmr_large"' in line or "key         = 'xlmr_large'" in line:
        # Walk backwards to find the dict( opening line
        # It's already in out — pop back to the dict( line
        while out and 'dict(' not in out[-1] and '# ── Text Model 1' not in out[-1]:
            out.pop()
        # Also remove the comment line before dict(
        if out and '# ── Text Model 1' in out[-1]:
            out.pop()
        # Now skip forward until we find the closing '),' of this dict
        depth = 1  # we're inside dict(
        # find the dict( in current context — already consumed, skip forward
        while i < len(lines):
            i += 1
            if i >= len(lines):
                break
            l = lines[i]
            depth += l.count('(') - l.count(')')
            if depth <= 0:
                i += 1  # skip the closing line too
                break
        continue

    out.append(line)
    i += 1

src2 = ''.join(out)

# Update title comments
src2 = src2.replace(
    '  DISASTER SEVERITY \u2013 ULTIMATE GRANDMASTER PIPELINE v5                      ',
    '  DISASTER SEVERITY \u2013 BANGLABERT-LARGE + VISION (Fast Variant)              '
).replace(
    '  3-Model Late Fusion: XLM-R-Large + BanglaBERT-Large + EfficientNet-B4    ',
    '  2-Model Late Fusion: BanglaBERT-Large + EfficientNet-B4                   '
).replace(
    '  Target: 0.88+ Weighted F1  |  Runtime Budget: < 8 hours on T4 GPU        ',
    '  Target: 0.83+ Weighted F1  |  Runtime Budget: ~3.5 hours on T4 GPU       '
).replace(
    '"submission_ultimate_v5.zip"',
    '"submission_banglabert_vision.zip"'
)

out_path = r'd:\Development\iiucdatathon\banglabert_vision_notebook.py'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(src2)

print(f'Written: {out_path}')
print(f'XLM-R still present: {"xlmr_large" in src2}')
print(f'BanglaBERT-Large present: {"banglabert_large" in src2}')
print(f'EfficientNet present: {"efficientnet" in src2}')
