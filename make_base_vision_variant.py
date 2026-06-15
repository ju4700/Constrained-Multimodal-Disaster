with open(r'd:\Development\iiucdatathon\ultimate_notebook.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out = []
skip = False
i = 0

# Remove XLM-R and replace BanglaBERT-Large with base BanglaBERT
new_cfgs = """MODEL_CFGS = [
    # ── Text Model: Base BanglaBERT (Fast, high batch size) ──────────────
    dict(
        key         = "banglabert_base",
        model_name  = "csebuetnlp/banglabert",
        type        = "text",
        epochs      = 5,
        lr          = 2e-5,
        batch       = 16,       # Base model fits easily
        grad_acc    = 2,        # Effective batch 32
        fp16        = True,
        grad_ckpt   = True,
        max_len     = 128,      
        use_rdrop   = True,     # We can afford R-Drop on base model
        rdrop_alpha = 0.30,
        use_awp     = False,
        awp_start   = 2,
        awp_lr      = 1e-4,
        awp_eps     = 1e-3,
        use_llrd    = True,
        llrd_decay  = 0.95,     # 0.95 is standard for 12-layer base models
        use_ema     = False,    # Disabled to prevent early-epoch score suppression
        ema_decay   = 0.999,
        label_smooth= 0.05,
        focal_gamma = 2.0,
    ),
"""

while i < len(lines):
    line = lines[i]
    if line.startswith("MODEL_CFGS = ["):
        out.append(new_cfgs)
        # Skip until the Vision Model dict
        while i < len(lines) and "── Vision Model:" not in lines[i]:
            i += 1
        # Also include the Vision Model line itself
        out.append(lines[i])
        i += 1
        continue
    
    # Remove adafactor (base model can use standard AdamW)
    if 'optim                       = "adafactor"' in line:
        i += 1
        continue
        
    out.append(line)
    i += 1

src2 = ''.join(out)

# Update titles
src2 = src2.replace(
    '  DISASTER SEVERITY \u2013 ULTIMATE GRANDMASTER PIPELINE v5                      ',
    '  DISASTER SEVERITY \u2013 BASE BANGLABERT + VISION                                '
).replace(
    '  3-Model Late Fusion: XLM-R-Large + BanglaBERT-Large + EfficientNet-B4    ',
    '  2-Model Late Fusion: Base BanglaBERT + EfficientNet-B4-NS                 '
).replace(
    '  Target: 0.88+ Weighted F1  |  Runtime Budget: < 8 hours on T4 GPU        ',
    '  Target: 0.81+ Weighted F1  |  Runtime Budget: < 2 hours on T4 GPU        '
).replace(
    '"submission_ultimate_v5.zip"',
    '"submission_base_vision.zip"'
)

out_path = r'd:\Development\iiucdatathon\base_vision_notebook.py'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(src2)

print(f'Written: {out_path}')
