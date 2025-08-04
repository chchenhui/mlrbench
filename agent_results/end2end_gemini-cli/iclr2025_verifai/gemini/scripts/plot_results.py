import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "experiment_results.csv")
PASS_RATE_PLOT = os.path.join(RESULTS_DIR, "pass_rate_comparison.png")
CONVERGENCE_PLOT = os.path.join(RESULTS_DIR, "convergence_rate.png")

def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"Results file not found at {RESULTS_CSV}. Run the experiment first.")
        return

    df = pd.read_csv(RESULTS_CSV)

    # --- 1. Plot Pass Rate Comparison ---
    pass_rate = df.groupby('method')['passed'].mean().reset_index()
    pass_rate['pass_rate'] = pass_rate['passed'] * 100

    plt.figure(figsize=(8, 6))
    sns.barplot(x='method', y='pass_rate', data=pass_rate, palette='viridis')
    
    plt.title('Comparison of Method Effectiveness (pass@1)')
    plt.xlabel('Method')
    plt.ylabel('Pass Rate (%)')
    plt.ylim(0, 100)
    for index, row in pass_rate.iterrows():
        plt.text(row.name, row.pass_rate + 2, f"{row.pass_rate:.1f}%", color='black', ha="center")

    plt.tight_layout()
    plt.savefig(PASS_RATE_PLOT)
    print(f"Pass rate plot saved to {PASS_RATE_PLOT}")
    plt.close()


    # --- 2. Plot Convergence Rate ---
    # This plot shows the percentage of problems solved within a certain number of iterations.
    convergence_data = []
    for method in ['UT-Repair', 'SMT-Repair']:
        method_df = df[(df['method'] == method) & (df['passed'] == True)]
        # Add solved problems at iteration 0 from Zero-Shot
        zero_shot_solved = df[(df['method'] == 'Zero-Shot') & (df['passed'] == True)]
        
        total_problems = df['task_id'].nunique()
        
        # Iteration 0 solved count
        solved_at_0 = len(zero_shot_solved)
        convergence_data.append({
            'method': method,
            'iteration': 0,
            'solved_rate': (solved_at_0 / total_problems) * 100
        })

        max_iter = df[df['method'] == method]['iteration'].max()
        for i in range(1, int(max_iter) + 1):
            solved_up_to_i = len(method_df[method_df['iteration'] <= i]) + solved_at_0
            rate = (solved_up_to_i / total_problems) * 100
            convergence_data.append({
                'method': method,
                'iteration': i,
                'solved_rate': rate
            })

    conv_df = pd.DataFrame(convergence_data)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=conv_df, x='iteration', y='solved_rate', hue='method', marker='o', palette='mako')

    plt.title('Convergence Rate of Repair Methods')
    plt.xlabel('Repair Iteration')
    plt.ylabel('Cumulative Pass Rate (%)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Method')
    plt.ylim(0, 105)
    plt.xticks(range(int(conv_df['iteration'].max()) + 1))

    plt.tight_layout()
    plt.savefig(CONVERGENCE_PLOT)
    print(f"Convergence plot saved to {CONVERGENCE_PLOT}")
    plt.close()


if __name__ == "__main__":
    main()
