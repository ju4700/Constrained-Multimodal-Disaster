import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

def visualize_benchmarks():
    print("="*60)
    print("DATASET BENCHMARKING RESULTS")
    print("="*60)

    # ──────────────────────────────────────────
    # Phase 1: BanglaCalamityMMD Ablation Study
    # ──────────────────────────────────────────
    if os.path.exists("phase1_results.csv"):
        print("\n[Phase 1] BanglaCalamityMMD Ablation Study:")
        df1 = pd.read_csv("phase1_results.csv")
        try: display(df1)
        except: print(df1.to_string(index=False))

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df1, x='Run', y='Test_Macro_F1', palette='viridis')
        plt.title('Phase 1: Ablation Study on BanglaCalamityMMD', fontsize=14, pad=15)
        plt.ylabel('Test Macro F1 Score', fontsize=12)
        plt.xlabel('Ablation Configuration', fontsize=12)
        plt.ylim(0.0, 1.0)
        for index, row in df1.iterrows():
            plt.text(index, row['Test_Macro_F1'] + 0.02, f"{row['Test_Macro_F1']:.4f}",
                     color='black', ha="center", fontweight='bold')
        plt.tight_layout()
        plt.savefig('phase1_ablation.png', dpi=300)
        plt.show()
    else:
        print("\n[Warning] phase1_results.csv not found.")

    # ──────────────────────────────────────────
    # Phase 2: CrisisMMD Generalization
    # ──────────────────────────────────────────
    if os.path.exists("phase2_results.csv"):
        print("\n[Phase 2] CrisisMMD Generalization Test:")
        df2 = pd.read_csv("phase2_results.csv")
        try: display(df2)
        except: print(df2.to_string(index=False))

        plt.figure(figsize=(6, 5))
        sns.barplot(data=df2, x='Phase', y='Macro_F1', palette='rocket')
        plt.title('Phase 2: Generalization to CrisisMMD v2.0', fontsize=14, pad=15)
        plt.ylabel('Test Macro F1 Score', fontsize=12)
        plt.ylim(0.0, 1.0)
        for index, row in df2.iterrows():
            plt.text(index, row['Macro_F1'] + 0.02, f"{row['Macro_F1']:.4f}",
                     color='black', ha="center", fontweight='bold')
        plt.tight_layout()
        plt.savefig('phase2_generalization.png', dpi=300)
        plt.show()
    else:
        print("\n[Warning] phase2_results.csv not found.")

    # ──────────────────────────────────────────
    # Phase 3: Unified Baseline Comparison
    # ──────────────────────────────────────────
    print("\n[Phase 3] Unified Baseline Comparison:")

    baseline_rows = []

    # Load text baselines
    if os.path.exists("baselines_text_results.csv"):
        baseline_rows.append(pd.read_csv("baselines_text_results.csv"))
    else:
        print("[Warning] baselines_text_results.csv not found.")

    # Load image baselines
    if os.path.exists("baselines_image_results.csv"):
        baseline_rows.append(pd.read_csv("baselines_image_results.csv"))
    else:
        print("[Warning] baselines_image_results.csv not found.")

    # Our best model (Run A from Phase 1)
    our_model = pd.DataFrame([{
        "Model":          "Ours: BanglaBERT + Swin-Tiny",
        "Modality":       "Multimodal",
        "Test Macro F1":  0.987,
        "Test Accuracy":  0.987
    }])
    baseline_rows.append(our_model)

    if not baseline_rows:
        print("[Error] No baseline CSVs found. Run baselines_text.py and baselines_image.py first.")
        return

    df_all = pd.concat(baseline_rows, ignore_index=True).drop_duplicates(subset='Model')
    df_all = df_all.sort_values("Test Macro F1", ascending=True)

    try: display(df_all)
    except: print(df_all.to_string(index=False))

    # ── Horizontal bar chart ──
    palette = {
        "Text-Only":   "#4e9af1",
        "Image-Only":  "#f4a261",
        "Multimodal":  "#2a9d8f"
    }
    colors = [palette.get(m, "#aaa") for m in df_all['Modality']]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(df_all['Model'], df_all['Test Macro F1'], color=colors, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, df_all['Test Macro F1']):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Test Macro F1 Score', fontsize=13)
    ax.set_title('BanglaCalamityMMD: Comprehensive Baseline Comparison', fontsize=14, pad=15)
    ax.set_xlim(0.0, 1.12)
    ax.axvline(x=df_all[df_all['Modality'] == 'Multimodal']['Test Macro F1'].max(),
               color='green', linestyle='--', linewidth=1.2, label='Our Model')

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in palette.items()]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=300)
    plt.show()
    print("\nSaved baseline_comparison.png")


if __name__ == "__main__":
    visualize_benchmarks()
