import os
import argparse
import subprocess
import csv

def find_experiments(base_root):
    return [d for d in os.listdir(base_root)
            if os.path.isdir(os.path.join(base_root, d))]

def run_logger(base_root, exp_name, fallback_clock_ns):
    full_dir = os.path.join(base_root, exp_name)
    print(f"\n🚀 Logging {exp_name} ...")

    result = subprocess.run([
        "python", "log_hls_results_to_wandb.py",
        "--run_name", exp_name,
        "--base_dir", full_dir,
        "--fallback_clock_ns", str(fallback_clock_ns)
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {exp_name} log 成功")
        return True, parse_metrics_from_output(result.stdout)
    else:
        print(f"❌ {exp_name} log 失敗")
        return False, {}

def parse_metrics_from_output(output_text):
    metrics = {}
    for line in output_text.splitlines():
        if ":" in line:
            try:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if val.replace('.', '', 1).isdigit():
                    metrics[key] = float(val)
            except:
                continue
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_root", type=str, required=True, help="Top-level folder containing all experiment folders")
    parser.add_argument("--fallback_clock_ns", type=float, default=5.0, help="Default clock period (ns)")
    args = parser.parse_args()

    experiments = find_experiments(args.base_root)
    print(f"🔍 發現 {len(experiments)} 個資料夾")

    success, fail = [], []
    summary_rows = []

    for exp in experiments:
        ok, metrics = run_logger(args.base_root, exp, args.fallback_clock_ns)
        if ok:
            success.append(exp)
            metrics["experiment"] = exp
            summary_rows.append(metrics)
        else:
            fail.append(exp)

    # Save summary.csv
    if summary_rows:
        keys = ["experiment"] + sorted({k for row in summary_rows for k in row})
        with open("summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print("📄 已生成 summary.csv")

    print("\n✅ 成功上傳：", len(success))
    for s in success:
        print("  -", s)

    print("\n❌ 失敗：", len(fail))
    for f in fail:
        print("  -", f)
