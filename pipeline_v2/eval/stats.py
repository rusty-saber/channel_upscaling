import os
import json
import numpy as np
from scipy import stats

def compute_ci(data, alpha=0.95, n_resamples=10000):
    if not data:
        return np.nan, np.nan
    res = stats.bootstrap((data,), np.mean, confidence_level=alpha, n_resamples=n_resamples, method='BCa')
    return res.confidence_interval.low, res.confidence_interval.high

def get_stats(data_list):
    if not data_list:
        return {}
    arr = np.array(data_list)
    return {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'median': np.median(arr),
        'iqr': np.percentile(arr, 75) - np.percentile(arr, 25),
        'min': np.min(arr),
        'max': np.max(arr)
    }

def load_data(base_path):
    results = {}
    for method in ['ssi', 'reve', 'zuna']:
        results[method] = {}
        for device in ['emotiv_epoc', 'muse_s', 'openbci_cyton']:
            results[method][device] = {'pearson': {}, 'mse': {}, 'beta_ratio': {}, 'bes': {}, 'beta_db': {}}
            
            per_subj_file = os.path.join(base_path, method, device, 'per_subject_results.json')
            bes_subj_file = os.path.join(base_path, method, device, 'bes_per_subject.json')
            
            if os.path.exists(per_subj_file):
                with open(per_subj_file, 'r') as f:
                    per_subj = json.load(f)
                    for subj, metrics_data in per_subj.items():
                        if metrics_data.get('status') == 'ok':
                            metrics = metrics_data['metrics']
                            results[method][device]['pearson'][subj] = metrics.get('pearson_mean')
                            # The MSE values in the JSON are in V^2 (e^-9 etc), paper uses uV^2
                            # 1 V = 1e6 uV -> 1 V^2 = 1e12 uV^2
                            results[method][device]['mse'][subj] = metrics.get('mse_mean') * 1e12
                            ratio = metrics.get('beta_mean_ratio')
                            results[method][device]['beta_ratio'][subj] = ratio
                            if ratio is not None and ratio > 0:
                                results[method][device]['beta_db'][subj] = 10 * np.log10(ratio)
                            
            if os.path.exists(bes_subj_file):
                with open(bes_subj_file, 'r') as f:
                    bes_subj = json.load(f)
                    for subj, val in bes_subj.items():
                        if isinstance(val, (int, float)):
                            results[method][device]['bes'][subj] = val
                        elif isinstance(val, dict) and 'bes' in val:
                            results[method][device]['bes'][subj] = val['bes']
    return results

def main():
    base_path = '../results'
    results = load_data(base_path)
    
    print("--- 95% CIs and Summaries ---")
    for method in ['ssi', 'reve', 'zuna']:
        for device in ['emotiv_epoc', 'muse_s', 'openbci_cyton']:
            print(f"\n{method.upper()} - {device}")
            for metric in ['pearson', 'mse', 'beta_ratio', 'beta_db', 'bes']:
                data = list(results[method][device][metric].values())
                # filter none or nan
                data = [x for x in data if x is not None and not np.isnan(x)]
                if data:
                    s = get_stats(data)
                    ci_low, ci_high = compute_ci(data)
                    print(f"  {metric}: mean={s['mean']:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}]), std={s['std']:.4f}, median={s['median']:.4f}, IQR={s['iqr']:.4f}, n={len(data)}")

    print("\n--- Wilcoxon Signed-Rank Tests (Device A - emotiv_epoc) ---")
    # ZUNA vs SSI and ZUNA vs REVE
    for device in ['emotiv_epoc']:
        for baseline in ['ssi', 'reve']:
            for metric in ['pearson', 'mse', 'beta_db', 'bes']:
                print(f"ZUNA vs {baseline.upper()} on {device} for {metric}:")
                subj_zuna = set(results['zuna'][device][metric].keys())
                subj_base = set(results[baseline][device][metric].keys())
                common = list(subj_zuna.intersection(subj_base))
                
                zuna_vals = [results['zuna'][device][metric][s] for s in common if results['zuna'][device][metric][s] is not None and not np.isnan(results['zuna'][device][metric][s])]
                base_vals = [results[baseline][device][metric][s] for s in common if results[baseline][device][metric][s] is not None and not np.isnan(results[baseline][device][metric][s])]
                
                if len(zuna_vals) > 1 and len(zuna_vals) == len(base_vals):
                    try:
                        stat, p = stats.wilcoxon(zuna_vals, base_vals)
                        print(f"  p-value: {p:.4e} (n={len(zuna_vals)})")
                    except Exception as e:
                        print(f"  Test failed: {e}")
                else:
                    print(f"  Not enough common subjects or mismatched lengths.")

if __name__ == '__main__':
    main()
