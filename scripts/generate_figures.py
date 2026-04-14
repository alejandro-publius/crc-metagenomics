import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

def fig1():
    bl = pd.read_csv('results/baseline_results.csv')
    cohorts = bl['cohort'].tolist()
    rf_joint = [0.790, 0.789, 0.714, 0.824, 0.716, 0.799, 0.823]
    xgb = pd.read_csv('results/joint_xgb_results.csv')['xgb_auc'].tolist()
    x = np.arange(len(cohorts))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, bl['auc'], w, label='Species RF', color='#2E86AB')
    ax.bar(x, rf_joint, w, label='Joint RF', color='#A23B72')
    ax.bar(x + w, xgb, w, label='Joint XGBoost', color='#F18F01')
    ax.set_ylabel('AUC-ROC')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_',' ') for c in cohorts], rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0.55, 0.95)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_title('LODO Cross-Validation: CRC vs Healthy', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig('figures/fig1_lodo_auc.png', dpi=300)
    plt.close()
    print('  fig1_lodo_auc.png')

def fig2():
    shap_df = pd.read_csv('results/shap_crc_features.csv').head(15).iloc[::-1]
    names = []
    for f in shap_df['feature']:
        parts = f.split('|')
        names.append(parts[-1].replace('s__','').replace('_',' ') if len(parts)>1 else f)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(shap_df)))[::-1]
    ax.barh(range(len(shap_df)), shap_df['mean_abs_shap'], color=colors)
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels(names, fontsize=8, style='italic')
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Top 15 Species Driving CRC Classification', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig('figures/fig2_shap_crc.png', dpi=300)
    plt.close()
    print('  fig2_shap_crc.png')

def fig3():
    ade = pd.read_csv('results/adenoma_results.csv')
    tasks = ['Healthy vs\nAdenoma', 'Adenoma vs\nCRC']
    x = np.arange(len(tasks))
    w = 0.3
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x - w/2, ade['rf_auc'], w, label='Random Forest', color='#2E86AB')
    ax.bar(x + w/2, ade['xgb_auc'], w, label='XGBoost', color='#F18F01')
    for i in range(len(tasks)):
        ax.text(i - w/2, ade['rf_auc'].iloc[i] + 0.01, f'{ade["rf_auc"].iloc[i]:.3f}', ha='center', fontsize=8)
        ax.text(i + w/2, ade['xgb_auc'].iloc[i] + 0.01, f'{ade["xgb_auc"].iloc[i]:.3f}', ha='center', fontsize=8)
    ax.set_ylabel('AUC-ROC')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0.5, 0.9)
    ax.legend()
    ax.set_title('Adenoma Classification (5-Fold CV)', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig('figures/fig3_adenoma.png', dpi=300)
    plt.close()
    print('  fig3_adenoma.png')

if __name__ == '__main__':
    print('Generating figures...')
    fig1()
    fig2()
    fig3()
    print('Done')
