import json, argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio not available - install with: pip install rasterio")

CLASS_MAP = {0:"NoData",1:"Marine Debris",2:"Dense Sargassum",3:"Sparse Sargassum",4:"Natural Organic",5:"Ship",6:"Clouds",7:"Marine Water",8:"Sediment-Laden Water",9:"Foam",10:"Turbid Water",11:"Shallow Water",12:"Waves",13:"Cloud Shadows",14:"Wakes",15:"Mixed Water"}
BAND_NAMES = ["B01s","B02","B03","B04","B05","B06s","B07s","B08","B8A","B11","B12"]
ZERO_PADDED = [0, 5, 6]
EXPECTED_SHAPE = (11, 256, 256)

def collect_patches(data_dir):
    patches_dir = data_dir / "patches"
    if not patches_dir.exists():
        return [], []
    imagery, labels = [], []
    for tif in sorted(patches_dir.rglob("*.tif")):
        stem = tif.stem
        if stem.endswith("_cl"):
            labels.append(tif)
        elif not stem.endswith("_conf"):
            imagery.append(tif)
    return imagery, labels

def phase1_structure(data_dir):
    patches_dir = data_dir / "patches"
    splits_dir = data_dir / "splits"
    scene_dirs = [d.name for d in sorted(patches_dir.iterdir()) if d.is_dir()] if patches_dir.exists() else []
    imagery, labels = collect_patches(data_dir)
    file_formats = set()
    if patches_dir.exists():
        for f in patches_dir.rglob("*"):
            if f.is_file():
                file_formats.add(f.suffix.lower())
    return {"data_dir":str(data_dir),"exists":data_dir.exists(),"patches_dir_exists":patches_dir.exists(),"splits_dir_exists":splits_dir.exists(),"scene_dirs":scene_dirs,"n_scenes":len(scene_dirs),"total_imagery_files":len(imagery),"total_label_files":len(labels),"file_formats":list(file_formats),"issues":[]}

def phase1b_inspect(data_dir, n_samples=10):
    imagery, labels = collect_patches(data_dir)
    if not imagery: return {"error":"No imagery","samples":[]}
    if not HAS_RASTERIO: return {"error":"rasterio missing","samples":[]}
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(imagery), size=min(n_samples,len(imagery)), replace=False)
    samples = []
    for idx in idxs:
        img = imagery[idx]
        lbl = img.parent/(img.stem+"_cl.tif")
        info = {"file":str(img.relative_to(data_dir))}
        try:
            with rasterio.open(img) as src:
                info.update({"bands":src.count,"height":src.height,"width":src.width,"dtype":src.dtypes[0]})
                data = src.read()
                info.update({"min":float(np.nanmin(data)),"max":float(np.nanmax(data)),"has_nan":bool(np.isnan(data).any()),"has_inf":bool(np.isinf(data).any()),"shape_ok":data.shape==EXPECTED_SHAPE,"dtype_ok":src.dtypes[0]=="float32","zero_bands_ok":all(np.allclose(data[b],0,atol=1e-6) for b in ZERO_PADDED if b<data.shape[0])})
        except Exception as e:
            info["error"] = str(e)
        if lbl.exists():
            try:
                with rasterio.open(lbl) as src:
                    lbl_data = src.read(1)
                    uniq = sorted(int(v) for v in np.unique(lbl_data))
                    info.update({"label_dtype":src.dtypes[0],"label_unique":uniq,"label_valid":all(0<=v<=15 for v in uniq)})
            except Exception as e:
                info["label_error"] = str(e)
        samples.append(info)
    return {"total_imagery":len(imagery),"total_labels":len(labels),"samples":samples}

def phase1c_splits(data_dir):
    splits_dir = data_dir / "splits"
    patches_dir = data_dir / "patches"
    if not splits_dir.exists():
        return {"error":"splits/ not found","splits":{},"issues":[],"duplicates":[]}
    result = {"splits":{},"issues":[],"duplicates":[]}
    all_patches = {}
    for pat, label in [("train*.txt","train"),("val*.txt","val"),("test*.txt","test")]:
        for sf in sorted(splits_dir.glob(pat)):
            names = [l.strip() for l in sf.read_text().strip().splitlines() if l.strip()]
            missing = []
            for pname in names:
                parts = pname.rsplit("_",1)
                scene = "S2_" + parts[0] if len(parts)==2 else "S2_" + pname
                if not (patches_dir/scene/f"S2_{pname}.tif").exists():
                    missing.append(pname)
            result["splits"][sf.stem] = {"file":sf.name,"count":len(names),"missing_count":len(missing),"missing_samples":missing[:5]}
            for pname in names:
                if pname in all_patches:
                    result["duplicates"].append({"patch":pname,"in":[all_patches[pname],sf.stem]})
                all_patches[pname] = sf.stem
    if result["duplicates"]:
        result["issues"].append(f"DATA LEAKAGE: {len(result['duplicates'])} patches in multiple splits!")
    return result

def phase2_distribution(data_dir):
    if not HAS_RASTERIO: return {"error":"rasterio missing"}
    patches_dir = data_dir / "patches"
    splits_dir = data_dir / "splits"
    if not patches_dir.exists(): return {"error":"patches/ not found"}
    result = {}
    for pat, sname in [("train*.txt","train"),("val*.txt","val"),("test*.txt","test")]:
        class_px = defaultdict(int)
        patch_presence = defaultdict(int)
        total_patches = patches_with_debris = nodata_total = 0
        for sf in sorted(splits_dir.glob(pat)) if splits_dir.exists() else []:
            for pname in [l.strip() for l in sf.read_text().strip().splitlines() if l.strip()]:
                parts = pname.rsplit("_",1)
                scene = "S2_" + parts[0] if len(parts)==2 else "S2_" + pname
                lbl_path = patches_dir/scene/f"S2_{pname}_cl.tif"
                if not lbl_path.exists(): continue
                total_patches += 1
                try:
                    with rasterio.open(lbl_path) as src:
                        lbl = src.read(1).astype(np.int32)
                    nodata_total += int(np.sum(lbl==0))
                    for cls, cnt in zip(*np.unique(lbl, return_counts=True)):
                        if cls == 0: continue
                        class_px[int(cls)] += int(cnt)
                        patch_presence[int(cls)] += 1
                    if 1 in np.unique(lbl): patches_with_debris += 1
                except: continue
        if not class_px: continue
        total = sum(class_px.values())
        pcts = {k:(v/total*100) for k,v in class_px.items()}
        vals = list(class_px.values())
        ratio = max(vals)/min(vals) if min(vals)>0 else float("inf")
        sev = "Extreme" if ratio>1000 else "Severe" if ratio>100 else "Moderate" if ratio>10 else "Mild"
        result[sname] = {"total_patches":total_patches,"patches_with_debris":patches_with_debris,"debris_patch_ratio":round(patches_with_debris/total_patches*100,2) if total_patches else 0,"total_labeled_pixels":total,"nodata_pixels":nodata_total,"class_pixels":{str(k):v for k,v in sorted(class_px.items())},"class_percentages":{str(k):round(v,6) for k,v in sorted(pcts.items())},"patch_class_presence":{str(k):v for k,v in sorted(patch_presence.items())},"imbalance_ratio":round(ratio,1),"severity":sev,"dominant_class":int(max(class_px,key=class_px.get)),"rarest_class":int(min(class_px,key=class_px.get)),"debris_pixels":class_px.get(1,0),"debris_percentage":round(pcts.get(1,0),6)}
    return result

def phase3_quality(data_dir, sample_size=200):
    if not HAS_RASTERIO: return {"error":"rasterio missing"}
    imagery, _ = collect_patches(data_dir)
    if not imagery: return {"error":"No imagery"}
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(imagery), size=min(sample_size,len(imagery)), replace=False)
    band_acc = {i:{"means":[],"stds":[],"mins":[],"maxs":[]} for i in range(11)}
    nan_c = inf_c = nd_c = 0
    zviol = {b:0 for b in ZERO_PADDED}
    flags = []
    checked = 0
    for idx in idxs:
        img = imagery[idx]
        checked += 1
        try:
            with rasterio.open(img) as src:
                data = src.read()
                nodata_val = src.nodata
            if np.isnan(data).any(): nan_c+=1; flags.append({"file":img.name,"issue":"NaN"})
            if np.isinf(data).any(): inf_c+=1; flags.append({"file":img.name,"issue":"Inf"})
            for b in ZERO_PADDED:
                if b<data.shape[0] and not np.allclose(data[b],0,atol=1e-6): zviol[b]+=1
            if nodata_val is not None:
                nd_px = int(np.sum(data[1]==nodata_val))
                if nd_px>0.5*256*256: nd_c+=1; flags.append({"file":img.name,"issue":f">50% nodata"})
            for b in range(min(data.shape[0],11)):
                bd = data[b].flatten()
                valid = bd[np.isfinite(bd)]
                if len(valid)>0:
                    band_acc[b]["means"].append(float(np.mean(valid)))
                    band_acc[b]["stds"].append(float(np.std(valid)))
                    band_acc[b]["mins"].append(float(np.min(valid)))
                    band_acc[b]["maxs"].append(float(np.max(valid)))
        except Exception as e:
            flags.append({"file":img.name,"issue":str(e)})
    band_stats = {}
    for b, acc in band_acc.items():
        if acc["means"]:
            band_stats[BAND_NAMES[b]] = {"global_mean":round(float(np.mean(acc["means"])),6),"global_std":round(float(np.mean(acc["stds"])),6),"global_min":round(float(np.min(acc["mins"])),6),"global_max":round(float(np.max(acc["maxs"])),6),"zero_padded":b in ZERO_PADDED}
    return {"samples_checked":checked,"nan_patches":nan_c,"inf_patches":inf_c,"high_nodata_patches":nd_c,"zero_band_violations":zviol,"band_statistics":band_stats,"quality_flags":flags[:60],"total_flags":len(flags)}

def phase4_labels(data_dir, sample_size=100):
    if not HAS_RASTERIO: return {"error":"rasterio missing"}
    _, labels = collect_patches(data_dir)
    if not labels: return {"error":"No labels"}
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(labels), size=min(sample_size,len(labels)), replace=False)
    invalid = []
    cpp = []
    co_occ = defaultdict(int)
    checked = 0
    for idx in idxs:
        lbl_path = labels[idx]
        checked += 1
        try:
            with rasterio.open(lbl_path) as src:
                lbl = src.read(1).astype(np.int32)
            uniq = [int(v) for v in np.unique(lbl)]
            non_nd = [v for v in uniq if v>0]
            if any(v<0 or v>15 for v in uniq): invalid.append(lbl_path.name)
            cpp.append(len(non_nd))
            for i,a in enumerate(non_nd):
                for b in non_nd[i+1:]:
                    co_occ[f"{min(a,b)}-{max(a,b)}"] += 1
        except: continue
    top_co = sorted(co_occ.items(), key=lambda x:x[1], reverse=True)[:10]
    return {"samples_checked":checked,"invalid_value_patches":invalid[:10],"total_invalid":len(invalid),"avg_classes_per_patch":round(float(np.mean(cpp)),2) if cpp else 0,"max_classes_per_patch":int(max(cpp)) if cpp else 0,"top_co_occurrences":[{"pair":k,"count":v} for k,v in top_co]}

def compute_weights(dist):
    if "train" not in dist: return {}
    tr = dist["train"]
    cp = tr.get("class_pixels",{})
    total = tr.get("total_labeled_pixels",0)
    if not cp or total==0: return {}
    w = {int(k):round(1.0/(v/total),4) for k,v in cp.items() if v>0}
    if w:
        mw = min(w.values())
        w = {k:round(v/mw,4) for k,v in w.items()}
    return w

def gen_report(struct,patches_info,splits,dist,qual,lq,weights,out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%H:%M:%S")
    rpt = out_dir/f"dataset_evaluation_{ds}.md"
    jsn = out_dir/"evaluation_summary.json"
    tr = dist.get("train",{})
    sev = tr.get("severity","?")
    dpct = tr.get("debris_percentage",0)
    ratio = tr.get("imbalance_ratio","?")
    L = []
    L.append(f"# Dataset Evaluation Report\n")
    L.append(f"- **Date**: {ds} {ts}")
    L.append(f"- **Dataset**: `{struct.get('data_dir','?')}`")
    L.append(f"- **Evaluator**: Dataset Evaluation Agent (ml_training_2.0)\n")
    L.append(f"## Executive Summary\n")
    L.append(f"MARIDA contains **{struct.get('total_imagery_files','?')} imagery patches** across **{struct.get('n_scenes','?')} scenes**. Class imbalance: **{sev}** ({ratio}:1). Marine Debris: **{dpct}%** of training pixels. Quality flags: {qual.get('total_flags',0)} across {qual.get('samples_checked',0)} patches.\n")
    L.append(f"## Structural Analysis\n")
    L.append(f"| Property | Value |"); L.append(f"|----------|-------|")
    L.append(f"| Imagery files | {struct.get('total_imagery_files',0)} |")
    L.append(f"| Label files | {struct.get('total_label_files',0)} |")
    L.append(f"| Scenes | {struct.get('n_scenes',0)} |")
    L.append(f"| File formats | {', '.join(struct.get('file_formats',[]))} |")
    L.append(f"| patches/ | {'YES' if struct.get('patches_dir_exists') else 'NO'} |")
    L.append(f"| splits/ | {'YES' if struct.get('splits_dir_exists') else 'NO'} |\n")
    if patches_info.get("samples"):
        L.append("### Patch Inspection\n")
        L.append("| File | Bands | Size | Dtype | Shape OK | ZeroBands | NaN | Inf |")
        L.append("|------|-------|------|-------|----------|-----------|-----|-----|")
        for s in patches_info["samples"]:
            if "error" not in s:
                L.append(f"| {s.get('file','?')} | {s.get('bands','?')} | {s.get('height','?')}x{s.get('width','?')} | {s.get('dtype','?')} | {'YES' if s.get('shape_ok') else 'NO'} | {'OK' if s.get('zero_bands_ok') else 'WARN'} | {'YES' if s.get('has_nan') else 'no'} | {'YES' if s.get('has_inf') else 'no'} |")
        L.append("")
    L.append("### Split Verification\n")
    L.append("| Split | Patches | Missing | Status |"); L.append("|-------|---------|---------|--------|")
    for n,i in splits.get("splits",{}).items():
        L.append(f"| {n} | {i['count']} | {i['missing_count']} | {'OK' if i['missing_count']==0 else 'WARN'} |")
    L.append("")
    if splits.get("duplicates"): L.append(f"**DATA LEAKAGE**: {len(splits['duplicates'])} patches in multiple splits!\n")
    L.append("## Class Distribution\n")
    for sn in ["train","val","test"]:
        sd = dist.get(sn)
        if not sd: continue
        L.append(f"### {sn.capitalize()} Split\n")
        L.append(f"- Patches: {sd['total_patches']} | With Debris: {sd['patches_with_debris']} ({sd['debris_patch_ratio']}%)")
        L.append(f"- Labeled pixels: {sd['total_labeled_pixels']:,} | Debris: {sd['debris_pixels']:,} ({sd['debris_percentage']}%)")
        L.append(f"- Imbalance: **{sd['severity']}** ({sd['imbalance_ratio']}:1) | Dominant: {CLASS_MAP.get(sd['dominant_class'],'?')} | Rarest: {CLASS_MAP.get(sd['rarest_class'],'?')}\n")
        L.append("| Cls | Name | Pixels | % | Patch Coverage |"); L.append("|-----|------|--------|---|----------------|")
        for k in sorted(sd["class_pixels"].keys(),key=int):
            px = sd["class_pixels"][k]; pct = sd["class_percentages"].get(k,0); cov = sd["patch_class_presence"].get(k,0)
            mark = " [TARGET]" if int(k)==1 else ""
            L.append(f"| {k} | {CLASS_MAP.get(int(k),'?')}{mark} | {px:,} | {pct:.4f}% | {cov}/{sd['total_patches']} |")
        L.append("")
    L.append("## Data Quality\n")
    L.append(f"| Metric | Value |"); L.append(f"|--------|-------|")
    L.append(f"| Checked | {qual.get('samples_checked',0)} |"); L.append(f"| NaN patches | {qual.get('nan_patches',0)} |")
    L.append(f"| Inf patches | {qual.get('inf_patches',0)} |"); L.append(f"| High nodata | {qual.get('high_nodata_patches',0)} |")
    L.append(f"| Total flags | {qual.get('total_flags',0)} |\n")
    zbv = qual.get("zero_band_violations",{})
    if zbv:
        L.append("### Zero-Padded Bands\n"); L.append("| Band | Violations | Status |"); L.append("|------|------------|--------|")
        for b in ZERO_PADDED:
            cnt = zbv.get(b,0); L.append(f"| {BAND_NAMES[b]} | {cnt} | {'OK' if cnt==0 else 'WARN'} |")
        L.append("")
    bs = qual.get("band_statistics",{})
    if bs:
        L.append("### Band Statistics\n"); L.append("| Band | ZeroPad | Mean | Std | Min | Max |"); L.append("|------|---------|------|-----|-----|-----|")
        for bn,st in bs.items():
            L.append(f"| {bn} | {'YES' if st.get('zero_padded') else 'no'} | {st['global_mean']} | {st['global_std']} | {st['global_min']} | {st['global_max']} |")
        L.append("")
    L.append("## Label Quality\n")
    L.append(f"- Checked: {lq.get('samples_checked',0)} | Invalid: {lq.get('total_invalid',0)}")
    L.append(f"- Avg classes/patch: {lq.get('avg_classes_per_patch',0)} | Max: {lq.get('max_classes_per_patch',0)}\n")
    co = lq.get("top_co_occurrences",[])
    if co:
        L.append("### Top Co-occurrences\n"); L.append("| Pair | Names | Count |"); L.append("|------|-------|-------|")
        for item in co:
            ids = item["pair"].split("-"); names=" + ".join(CLASS_MAP.get(int(i),f"#{i}") for i in ids)
            L.append(f"| {item['pair']} | {names} | {item['count']} |")
        L.append("")
    L.append("## Diagnosis\n")
    L.append(f"1. **CRITICAL - Extreme imbalance**: Debris={dpct}% ({ratio}:1)")
    L.append(f"2. **CRITICAL - Zero-padded bands**: B01,B06,B07 always zero (3/11 channels)")
    L.append(f"3. **MODERATE - Small dataset**: {struct.get('total_imagery_files','?')} patches requires heavy augmentation\n")
    L.append("## Recommended Solutions\n")
    L.append("| Solution | Priority |"); L.append("|----------|----------|")
    L.append("| Oversampling debris patches 3-5x | High |")
    L.append("| Focal Loss (gamma=2.0, alpha=0.25) | High |")
    L.append("| **COMBINED: 2x oversampling + 0.5*Focal + 0.5*Dice** | BEST |")
    L.append("| Heavy augmentation on debris patches | Medium |\n")
    if weights:
        L.append("### Class Weights (inverse freq, normalized)\n"); L.append("| Class | Name | Weight |"); L.append("|-------|------|--------|")
        for cid in sorted(weights.keys()): L.append(f"| {cid} | {CLASS_MAP.get(cid,f'#{cid}')} | {weights[cid]} |")
        L.append("")
    L.append("## Training Recommendations (ml_training_2.0)\n")
    L.append("| Decision | Recommendation |"); L.append("|----------|---------------|")
    L.append("| Architecture | SegFormer-B2 (primary) or U-Net ResNet-34 (baseline) |")
    L.append("| Loss | 0.5*Focal(g=2)+0.5*Dice |")
    L.append("| Optimizer | AdamW lr=1e-4 wd=1e-4 |")
    L.append("| Scheduler | CosineAnnealingLR |")
    L.append("| Batch size | 8-16 (RTX 4050 6GB VRAM) |")
    L.append("| Primary metric | Debris IoU (not mIoU!) |")
    L.append("| Early stopping | patience=15 on val_debris_iou |")
    rpt.write_text("\n".join(L), encoding="utf-8")
    summary = {"date":ds,"dataset_dir":struct.get("data_dir"),"total_imagery":struct.get("total_imagery_files",0),"total_labels":struct.get("total_label_files",0),"n_scenes":struct.get("n_scenes",0),"split_counts":{k:v["count"] for k,v in splits.get("splits",{}).items()},"imbalance":{s:{"severity":d["severity"],"ratio":d["imbalance_ratio"],"debris_pct":d["debris_percentage"]} for s,d in dist.items() if isinstance(d,dict) and "severity" in d},"class_weights":weights,"quality":{"nan":qual.get("nan_patches",0),"inf":qual.get("inf_patches",0),"flags":qual.get("total_flags",0)},"band_stats":qual.get("band_statistics",{})}
    with open(jsn,"w") as f: json.dump(summary,f,indent=2,default=str)
    return rpt, jsn

def main():
    parser = argparse.ArgumentParser(description="MARIDA Evaluation ml_training_2.0")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=200)
    args = parser.parse_args()
    dd = Path(args.data_dir).resolve()
    od = Path(args.output_dir).resolve()
    print("="*65)
    print("  MARIDA DATASET EVALUATION - ml_training_2.0")
    print("="*65)
    print(f"  Dataset: {dd}")
    print(f"  Output:  {od}")
    print(f"  rasterio: {'OK' if HAS_RASTERIO else 'MISSING'}")
    print("="*65)
    from datetime import datetime; t0=datetime.now()
    print("\n[Phase 1] Structure..."); s=phase1_structure(dd); print(f"  -> {s['total_imagery_files']} imgs | {s['total_label_files']} lbls | {s['n_scenes']} scenes")
    print("\n[Phase 1b] Patch inspection..."); pi=phase1b_inspect(dd,10); print(f"  -> {len(pi.get('samples',[]))} samples")
    print("\n[Phase 1c] Splits..."); sp=phase1c_splits(dd)
    for n,i in sp.get("splits",{}).items(): print(f"  {n}: {i['count']} [{('OK' if i['missing_count']==0 else 'WARN')}]")
    print("\n[Phase 2] Class distribution (may take minutes)..."); dist=phase2_distribution(dd)
    for sn,sd in dist.items():
        if isinstance(sd,dict) and "severity" in sd: print(f"  {sn}: {sd['severity']} ({sd['imbalance_ratio']}:1) debris={sd['debris_percentage']}%")
    print(f"\n[Phase 3] Data quality ({args.sample_size} patches)..."); q=phase3_quality(dd,args.sample_size); print(f"  -> NaN:{q.get('nan_patches',0)} Inf:{q.get('inf_patches',0)} Flags:{q.get('total_flags',0)}")
    print("\n[Phase 4] Label quality..."); lq=phase4_labels(dd,100); print(f"  -> avg classes/patch:{lq.get('avg_classes_per_patch',0)}")
    w=compute_weights(dist); print(f"\n[Weights] Marine Debris weight: {w.get(1,'N/A')}")
    print("\n[Report] Generating..."); rpt,jsn=gen_report(s,pi,sp,dist,q,lq,w,od)
    elapsed=(datetime.now()-t0).total_seconds()
    print(f"\n{'='*65}\n  Done in {elapsed:.1f}s\n  Report: {rpt}\n  JSON:   {jsn}\n{'='*65}")

if __name__ == "__main__":
    main()
