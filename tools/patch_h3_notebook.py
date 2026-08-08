#!/usr/bin/env python3
"""Wire the MiniMax H3 Director acceleration module into the Colab notebook.

Upstream (AIMixer/ComfyUI_MiniMaxH3_Director) added
example_workflows/minimax_h3_director_加速版.json on 2026-08-07. The speedup is
not a new model or a LoRA: it chains two KJNodes MODEL patchers in front of the
Director node.

    UNETLoader
      -> PathchSageAttentionKJ                        (sage_attention=auto)
      -> MiniMaxH3MemoryEfficientSageAttentionPatch
      -> MiniMaxH3Director.model

This script patches the notebook so Cell 4 installs KJNodes + sageattention and
Cell 6 injects those two nodes into the embedded workflow. All edits are
anchored string replacements on individual cell sources, so the gzip+base64
workflow blob in Cell 2 is never touched.

    python tools/patch_h3_notebook.py
    python tools/patch_h3_notebook.py --check
    python tools/patch_h3_notebook.py --nb "MiniMax_H3_Director_Colab_Pro_v4.ipynb"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_NB = "MiniMax_H3_Director_Colab_Pro_v4 (2).ipynb"
MARKER = "accel_sage"


# --------------------------------------------------------------------------
# Cell 1 - configuration
# --------------------------------------------------------------------------

CFG_OLD = """    # --- 其他 ---
    "sage_mode": "skip",       # 本工作流里没有 Patch Sage Attention 节点，默认不装
"""

CFG_NEW = """    # --- 加速模块（对应导演台 2026-08-07 新增的「加速版」示例工作流）---
    #   UNETLoader -> PathchSageAttentionKJ
    #              -> MiniMaxH3MemoryEfficientSageAttentionPatch -> 导演台
    #   两个节点都来自 KJNodes；装不上 sageattention 时会以旁路状态插入，不影响出片。
    "accel_sage": True,
    "accel_steps": 0,          # >0 就把导演台 steps 改成这个值（加速版示例是 20）

    # --- 其他 ---
    "sage_mode": "auto",       # auto = 自动装 sageattention；skip = 不装、也不插加速节点
"""


# --------------------------------------------------------------------------
# Cell 4 - custom nodes and sageattention
# --------------------------------------------------------------------------

REPO_OLD = """    "PathchSageAttentionKJ": "https://github.com/kijai/ComfyUI-KJNodes.git",
"""

REPO_NEW = """    "PathchSageAttentionKJ": "https://github.com/kijai/ComfyUI-KJNodes.git",
    "MiniMaxH3MemoryEfficientSageAttentionPatch": "https://github.com/kijai/ComfyUI-KJNodes.git",
"""

FORCE_OLD = """if CFG["use_turbo_lora"]:
    repos.append("https://github.com/shuaixn/ComfyUI-MiniMaxH3DualClockSampler.git")
"""

FORCE_NEW = """#   加速模块的两个节点不在内嵌工作流里，wf_types 反推不到，这里显式补上 KJNodes
if CFG.get("accel_sage") and CFG["sage_mode"] != "skip":
    kj = NODE_REPOS["PathchSageAttentionKJ"]
    if kj not in repos:
        repos.append(kj)
        log("加速模块需要 KJNodes -> ComfyUI-KJNodes")
if CFG["use_turbo_lora"]:
    repos.append("https://github.com/shuaixn/ComfyUI-MiniMaxH3DualClockSampler.git")
"""

SAGE_START = "# ---------- Sage Attention（本工作流用不到，默认跳过）----------\n"
SAGE_END = "# ---------- ComfyUI-Manager 离线模式"

SAGE_NEW = """# ---------- Sage Attention（加速模块的依赖）----------
#   PathchSageAttentionKJ 把注意力后端换成 sageattn；
#   MiniMaxH3MemoryEfficientSageAttentionPatch 换掉 H3 自注意力实现以压低峰值显存。
#   后者在 KJNodes 里标 EXPERIMENTAL，支持 sm80/86/89/90/120（A100 sm_80、L4 sm_89 都在内）。
SAGE_PROBE = "import sageattention as s;print('SAGE_OK=' + getattr(s,'__version__','1.x'))"


def sage_version():
    out = sh('%s -c "%s"' % (sys.executable, SAGE_PROBE), check=False, quiet=True)
    for line in out.splitlines():
        if line.startswith("SAGE_OK="):
            return line[8:].strip()
    return None


if CFG["sage_mode"] == "skip":
    CFG["sage_ok"], CFG["sage_ver"] = False, ""
    log("sage_mode=skip：不装 sageattention，Cell 6 也不会插加速节点")
else:
    ver = sage_version()
    if not ver:
        log("安装 sageattention（纯 Triton 轮子，不会动 torch）...")
        pip("--no-deps sageattention")
        ver = sage_version() or ""
    if not ver:
        pip("--no-deps sageattention==1.0.6")
        ver = sage_version() or ""
    CFG["sage_ok"], CFG["sage_ver"] = bool(ver), ver
    if ver:
        log("sageattention %s 可用" % ver)
        if ver.split(".")[0] == "1":
            log("PyPI 上只有 1.x：PatchSageAttentionKJ 能用；"
                "MiniMaxH3MemoryEfficientSageAttentionPatch 要 2.x 内核，"
                "启动后若报错就在界面里选中它按 Ctrl+B 旁路", "!")
    else:
        log("sageattention 装不上，加速节点会以旁路状态插入，不影响正常出片", "!")
    if CFG.get("sm") and CFG["sm"] not in (80, 86, 89, 90, 120):
        log("sm_%d 不在加速节点的支持列表（80/86/89/90/120）里" % CFG["sm"], "!")

"""


# --------------------------------------------------------------------------
# Cell 6 - inject the two patch nodes into the workflow graph
# --------------------------------------------------------------------------

INJECT_TAIL = """wf_save(wf)

# ---------- 装进 ComfyUI（侧栏直接能打开）----------"""

INJECT_NEW = '''# ---------- 加速模块：UNETLoader -> Sage 补丁 x2 -> 导演台 ----------
#   对应 AIMixer/ComfyUI_MiniMaxH3_Director 的
#   example_workflows/minimax_h3_director_加速版.json（作者 2026-08-07 新增）
ACCEL_CHAIN = [
    ("PathchSageAttentionKJ", "Patch Sage Attention KJ",
     ["auto", False], [270, 82], "MODEL"),
    ("MiniMaxH3MemoryEfficientSageAttentionPatch",
     "MiniMax H3 Mem Eff Sage Attention Patch", [], [330, 26], "model"),
]


def model_slot(node):
    return next((i for i, inp in enumerate(node.get("inputs", []))
                 if inp.get("name") == "model"), None)


def insert_accel(wf, d):
    """把补丁节点逐个串到上游与导演台之间（挂了 Turbo LoRA 就接在 LoRA 后面）。"""
    out = []
    mode = 0 if CFG.get("sage_ok") else 4      # 4 = 旁路，界面里 Ctrl+B 可开关
    for i, (ntype, title, widgets, size, oname) in enumerate(ACCEL_CHAIN):
        if wf_find(wf, ntype):
            continue
        slot = model_slot(d)
        if slot is None:
            out.append("导演台没有 model 输入口，跳过加速模块")
            break
        cur = d["inputs"][slot].get("link")
        old = next((l for l in wf["links"] if l[0] == cur), None)
        if old is None:
            out.append("找不到导演台的 model 连线，跳过加速模块")
            break
        nid = max(n["id"] for n in wf["nodes"]) + 1
        lid = max([l[0] for l in wf["links"]] + [wf.get("last_link_id", 0)]) + 1
        wf["nodes"].append({
            "id": nid, "type": ntype, "title": title,
            "pos": [d["pos"][0] + 60 * i, d["pos"][1] - 240 + 80 * i],
            "size": size, "flags": {}, "order": 5, "mode": mode,
            "inputs": [{"name": "model", "type": "MODEL", "link": old[0]}],
            "outputs": [{"name": oname, "type": "MODEL",
                         "links": [lid], "slot_index": 0}],
            "properties": {"Node name for S&R": ntype},
            "widgets_values": list(widgets),
        })
        old[3], old[4] = nid, 0                # 上游 -> 新节点
        wf["links"].append([lid, nid, 0, d["id"], slot, "MODEL"])
        d["inputs"][slot]["link"] = lid        # 新节点 -> 导演台
        wf["last_link_id"] = lid
        wf["last_node_id"] = max(wf.get("last_node_id", 0), nid)
        out.append("插入 %s%s" % (ntype, "" if mode == 0 else "（旁路）"))
    return out


if d and CFG.get("accel_sage") and CFG["sage_mode"] != "skip":
    changes += insert_accel(wf, d)
    if CFG.get("accel_steps"):
        j = widget_after(d, "高级采样", int)
        if j >= 0 and d["widgets_values"][j] != CFG["accel_steps"]:
            d["widgets_values"][j] = CFG["accel_steps"]
            changes.append("steps -> %d（加速版）" % CFG["accel_steps"])

'''


# --------------------------------------------------------------------------
# Cell 8 - do not fail the health check on bypassed nodes
# --------------------------------------------------------------------------

CHECK_OLD = """    for t in wf_types(wf):
        if t in ("Note", "MarkdownNote", "Reroute"):
            continue
"""

CHECK_NEW = """    bypassed = {n.get("type") for n in wf["nodes"] if n.get("mode") == 4}
    for t in wf_types(wf):
        if t in ("Note", "MarkdownNote", "Reroute") or t in bypassed:
            continue          # 旁路(Ctrl+B)的节点不参与体检
"""


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

DOC_OLD = (
    "- `sage_mode = \"triton\"` 可以装 SageAttention，"
    "但本工作流没有 `Patch Sage Attention KJ` 节点，装了也不会生效。\n"
)

DOC_NEW = (
    "- **加速模块**：`accel_sage = True`（默认）会让 Cell 4 装好 KJNodes + sageattention，"
    "Cell 6 自动把 `UNETLoader → PathchSageAttentionKJ → "
    "MiniMaxH3MemoryEfficientSageAttentionPatch → 导演台` 串起来，"
    "对应作者 2026-08-07 新增的 `minimax_h3_director_加速版.json`。\n"
    "- sageattention 装不上时，两个节点仍会插入但处于**旁路**状态，"
    "选中按 `Ctrl+B` 即可开关，出片不受影响。\n"
    "- 加速是近似计算（Q/K 量化到 int8、V 到 fp8），画面与不开时会有细微差别；"
    "要严格做同 seed 比对就把它旁路掉。\n"
    "- 不想要就把 `accel_sage` 设成 `False`，或把 `sage_mode` 设成 `\"skip\"`。\n"
)

TABLE_OLD = (
    "| 换了 r2v / v2v / rv2v 任务报形状错 | 这些任务要 `ref2va` 底模，你还挂着 fl2va | "
    "两套底模默认都已在本地：把 UNETLoader 换成 `minimax_h3_ref2va_*` 那个文件即可，"
    "或重跑 Cell 2 → 6 自动回写 |\n"
)

TABLE_NEW = TABLE_OLD + (
    "| 加速节点报 `sageattention ... required` 或内核错误 | "
    "PyPI 上的 sageattention 是 1.x，`MiniMaxH3MemoryEfficientSageAttentionPatch` 要 2.x 内核 | "
    "选中该节点 `Ctrl+B` 旁路，只留 `Patch Sage Attention KJ`；或自行源码编译 SageAttention 2 |\n"
    "| 开了加速后画面和之前不一样 | 加速走的是近似注意力（int8/fp8 量化） | "
    "正常现象；要同 seed 严格比对就把加速模块旁路 |\n"
)


OPS = [
    {"name": "Cell 1: accel_sage / accel_steps / sage_mode=auto",
     "kind": "replace", "old": CFG_OLD, "new": CFG_NEW},
    {"name": "Cell 4: map the patch node to KJNodes",
     "kind": "replace", "old": REPO_OLD, "new": REPO_NEW},
    {"name": "Cell 4: always install KJNodes when accel is on",
     "kind": "replace", "old": FORCE_OLD, "new": FORCE_NEW},
    {"name": "Cell 4: install and probe sageattention",
     "kind": "range", "start": SAGE_START, "end": SAGE_END, "new": SAGE_NEW},
    {"name": "Cell 6: inject the two patch nodes into the graph",
     "kind": "replace", "old": INJECT_TAIL, "new": INJECT_NEW + INJECT_TAIL},
    {"name": "Cell 8: skip bypassed nodes in the health check",
     "kind": "replace", "old": CHECK_OLD, "new": CHECK_NEW},
    {"name": "Docs: acceleration notes",
     "kind": "replace", "old": DOC_OLD, "new": DOC_NEW},
    {"name": "Docs: troubleshooting rows",
     "kind": "replace", "old": TABLE_OLD, "new": TABLE_NEW},
]


def replace_once(src, old, new, name):
    hits = src.count(old)
    if hits != 1:
        sys.exit("anchor matched %d times, expected 1: %s" % (hits, name))
    return src.replace(old, new)


def cut_between(src, start, end, new, name):
    i = src.find(start)
    j = src.find(end, i + 1) if i >= 0 else -1
    if i < 0 or j < 0:
        sys.exit("range anchor missing: " + name)
    return src[:i] + new + src[j:]


def apply_ops(nb):
    applied = []
    for op in OPS:
        placed = False
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            probe = op["start"] if op["kind"] == "range" else op["old"]
            if probe not in src:
                continue
            if op["kind"] == "range":
                src = cut_between(src, op["start"], op["end"], op["new"], op["name"])
            else:
                src = replace_once(src, op["old"], op["new"], op["name"])
            cell["source"] = src.splitlines(keepends=True)
            applied.append(op["name"])
            placed = True
            break
        if not placed:
            sys.exit("anchor not found: " + op["name"])
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch the H3 Director Colab notebook for the accel module."
    )
    parser.add_argument("--nb", default=DEFAULT_NB, help="notebook path, repo-relative")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the notebook has not been patched yet",
    )
    args = parser.parse_args()

    path = ROOT / args.nb
    if not path.exists():
        sys.exit("notebook not found: " + str(path))

    nb = json.loads(path.read_text(encoding="utf-8"))
    if any(MARKER in "".join(c["source"]) for c in nb["cells"]):
        print("already patched: " + args.nb)
        return
    if args.check:
        sys.exit("NOT patched: " + args.nb)

    applied = apply_ops(nb)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("patched " + args.nb)
    for name in applied:
        print("  - " + name)


if __name__ == "__main__":
    main()
