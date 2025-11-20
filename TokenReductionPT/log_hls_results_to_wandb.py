import argparse
import xml.etree.ElementTree as ET
import re
import os
import wandb
import sys

def check_file(path, label):
    if os.path.exists(path):
        print(f"✅ 找到 {label}")
        return True
    else:
        print(f"❌ 缺少 {label}: {path}")
        return False

def parse_csynth(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    latency_node = root.find(".//PerformanceEstimates/SummaryOfOverallLatency/Average-caseRealTimeLatency")
    latency_ms = float(latency_node.text.replace(" ms", "")) if latency_node is not None else None

    clock_node = root.find(".//UserAssignments/TargetClockPeriod")
    clock_ns = float(clock_node.text) if clock_node is not None else None

    resources_node = root.find(".//AreaEstimates/Resources")
    res = {elem.tag: int(elem.text) for elem in resources_node}
    avail_node = root.find(".//AreaEstimates/AvailableResources")
    avail = {elem.tag: int(elem.text) for elem in avail_node}

    return {
        "latency_csynth": latency_ms,
        "clock_ns": clock_ns,
        "LUT_csynth": res.get("LUT"),
        "LUT_csynth_util": res.get("LUT") / avail.get("LUT") * 100 if avail.get("LUT") else None,
        "DSP_csynth": res.get("DSP"),
        "DSP_csynth_util": res.get("DSP") / avail.get("DSP") * 100 if avail.get("DSP") else None,
        "BRAM_18K_csynth": res.get("BRAM_18K"),
        "BRAM_18K_csynth_util": res.get("BRAM_18K") / avail.get("BRAM_18K") * 100 if avail.get("BRAM_18K") else None,
        "URAM_csynth": res.get("URAM"),
        "URAM_csynth_util": res.get("URAM") / avail.get("URAM") * 100 if avail.get("URAM") else None,
    }


def parse_synth_rpt(path):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    def extract(kw):
        m = re.search(rf"\|\s*{kw}\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([\d\.]+)", text)
        return (int(m.group(1)), float(m.group(2))) if m else (None, None)

    lut, lut_util = extract("CLB LUTs\\*")
    bram, bram_util = extract("Block RAM Tile")
    uram, uram_util = extract("URAM")
    dsp, dsp_util = extract("DSPs")

    return {
        "LUT_vsynth": lut,
        "LUT_vsynth_util": lut_util,
        "DSP_vsynth": dsp,
        "DSP_vsynth_util": dsp_util,
        "BRAM_18K_vsynth": bram,
        "BRAM_18K_vsynth_util": bram_util,
        "URAM_vsynth": uram,
        "URAM_vsynth_util": uram_util,
    }


def parse_power_rpt(path):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    m = re.search(r"Total On-Chip Power \(W\)\s*\|\s*([\d\.]+)", text)
    return {"power_w": float(m.group(1))} if m else {}

def parse_cosim(path, clock_ns):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    m = re.search(r"\|\s*Verilog\|\s*Pass\|\s*(\d+)\|", text)
    if m:
        cycles = int(m.group(1))
        latency_ms = cycles * clock_ns / 1e6
        return {"latency_vsynth": latency_ms}
    return {}

def log_to_wandb(metrics, run_name):
    if not metrics or all(v is None for v in metrics.values()):
        print(f"⏩ 略過 {run_name}，沒有任何數據可 log")
        return

    wandb.init(project="TR_FPGA", name=run_name)
    wandb.log(metrics)
    wandb.finish()
    print(f"✅ 已整合並上傳至 wandb: {run_name}")


def auto_paths(base_dir):
    return {
        "csynth_xml": os.path.join(base_dir, "myproject_prj/solution1/syn/report/myproject_csynth.xml"),
        "synth_rpt": os.path.join(base_dir, "vivado_synth.rpt"),
        "power_rpt": os.path.join(base_dir, "ex1_post_route_design.pwr"),
        "cosim_rpt": os.path.join(base_dir, "myproject_prj/solution1/sim/report/myproject_cosim.rpt")
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--fallback_clock_ns", type=float, default=5.0)
    args = parser.parse_args()

    paths = auto_paths(args.base_dir)
    metrics = {}
    clock_ns = args.fallback_clock_ns

    # csynth
    if check_file(paths["csynth_xml"], "csynth.xml"):
        try:
            result = parse_csynth(paths["csynth_xml"])
            clock_ns = result.pop("clock_ns") or clock_ns
            metrics.update(result)
        except Exception as e:
            print("⚠️ csynth 解析失敗：", e)

    # synth
    if check_file(paths["synth_rpt"], "vivado_synth.rpt"):
        try:
            metrics.update(parse_synth_rpt(paths["synth_rpt"]))
        except Exception as e:
            print("⚠️ synth.rpt 解析失敗：", e)

    # power
    if check_file(paths["power_rpt"], "post_route_design.pwr"):
        try:
            metrics.update(parse_power_rpt(paths["power_rpt"]))
        except Exception as e:
            print("⚠️ power.rpt 解析失敗：", e)

    # cosim
    if check_file(paths["cosim_rpt"], "myproject_cosim.rpt"):
        try:
            result = parse_cosim(paths["cosim_rpt"], clock_ns)
            metrics.update(result)
            if "latency_vsynth" in result and "power_w" in metrics:
                energy = metrics["power_w"] * result["latency_vsynth"] / 1000
                metrics["energy_j"] = energy
        except Exception as e:
            print("⚠️ cosim.rpt 解析失敗：", e)

    if metrics and any(v is not None for v in metrics.values()):
        print(f"📤 Log 成功，內容包含 {len(metrics)} 筆欄位")
        print(f"📌 使用 clock_ns = {clock_ns} ns")
        log_to_wandb(metrics, args.run_name)
        sys.exit(0)
    else:
        print("⚠️ 沒有任何有效數據可記錄，略過該 run")
        sys.exit(1)
