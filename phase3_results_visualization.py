import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_benchmarks():
    print("="*50)
    print("🔥 DATASET BENCHMARKING RESULTS 🔥")
    print("="*50)
    
    # 1. Load Phase 1 Results
    if os.path.exists("phase1_results.csv"):
        print("\n[Phase 1] BanglaCalamityMMD Ablation Study:")
        df1 = pd.read_csv("phase1_results.csv")
        display(df1) # Kaggle automatically renders this nicely
        
        # Plot Phase 1
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df1, x='Run', y='Test_Macro_F1', palette='viridis')
        plt.title('Phase 1: Ablation Study on BanglaCalamityMMD', fontsize=14, pad=15)
        plt.ylabel('Test Macro F1 Score', fontsize=12)
        plt.xlabel('Ablation Configuration', fontsize=12)
        plt.ylim(0.0, 1.0)
        
        # Add text labels on top of bars
        for index, row in df1.iterrows():
            plt.text(index, row['Test_Macro_F1'] + 0.02, f"{row['Test_Macro_F1']:.4f}", 
                     color='black', ha="center", fontweight='bold')
            
        plt.tight_layout()
        plt.show()
    else:
        print("\n⚠️ phase1_results.csv not found! Make sure Phase 1 finished running.")

    # 2. Load Phase 2 Results
    if os.path.exists("phase2_results.csv"):
        print("\n[Phase 2] CrisisMMD Generalization Test:")
        df2 = pd.read_csv("phase2_results.csv")
        display(df2)
        
        # Plot Phase 2
        plt.figure(figsize=(6, 5))
        sns.barplot(data=df2, x='Phase', y='Macro_F1', palette='rocket')
        plt.title('Phase 2: Generalization to CrisisMMD_v2.0', fontsize=14, pad=15)
        plt.ylabel('Test Macro F1 Score', fontsize=12)
        plt.ylim(0.0, 1.0)
        
        for index, row in df2.iterrows():
            plt.text(index, row['Macro_F1'] + 0.02, f"{row['Macro_F1']:.4f}", 
                     color='black', ha="center", fontweight='bold')
            
        plt.tight_layout()
        plt.show()
    else:
        print("\n⚠️ phase2_results.csv not found! Make sure Phase 2 finished running.")

if __name__ == "__main__":
    visualize_benchmarks()
