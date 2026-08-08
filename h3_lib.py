# -*- coding: utf-8 -*-
# MiniMax-H3 导演台 Colab 运行版的公共小工具。
# 由 notebook 的 Cell 1 拉下来 exec 进全局命名空间，用法和写在格子里完全一样。
# 这些函数几乎不需要改，抽出来是为了让 notebook 只剩下配置和流程。
import json, os, shutil, struct, subprocess, sys, time
import re

def save_cfg():
    with open(CFG["cfg_path"], "w") as f:
        json.dump(CFG, f, indent=2, ensure_ascii=False)


DERIVED = ("sm", "cap", "vram_gb", "gpu_name", "tier", "can_fp8", "can_nvfp4",
           "dit_file", "dit_files", "te_file", "vae_files", "lora_ready",
           "sage_ok", "sage_ver", "sage_kernel",
           "wf_types", "assets", "task_type", "model_family", "workflow_installed")


def load_cfg():
    """只重跑 Cell 1 + Cell 8 时，把前面几格探测/下载的结果捞回来。"""
    if os.path.exists(CFG["cfg_path"]):
        old = json.load(open(CFG["cfg_path"]))
        for k in DERIVED:
            if old.get(k) not in (None, "", [], {}) and not CFG.get(k):
                CFG[k] = old[k]
    return CFG


def log(msg, tag="*"):
    print("[%s] %s" % (tag, msg), flush=True)


def sh(cmd, cwd=None, check=True, quiet=False):
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.stdout and not quiet:
        print(r.stdout.strip()[-4000:], flush=True)
    if check and r.returncode != 0:
        raise RuntimeError("命令失败(%d): %s" % (r.returncode, cmd))
    return r.stdout or ""


def pip(pkgs):
    sh("%s -m pip install -q %s" % (sys.executable, pkgs), quiet=True)


def free_gb(path=None):
    return shutil.disk_usage(path or ROOT).free / 1024 ** 3


def ram_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return 0.0


# ---------- 工作流读写小工具（Cell 2 / 6 / 7 / 8 全都用它们，所以放在最前面）----------
LOADER_FIELDS = {          # 节点类型 -> (widget 下标, models 子目录)
    "UNETLoader": (0, "diffusion_models"),
    "CLIPLoader": (0, "text_encoders"),
    "VAELoader": (0, "vae"),
    "LoraLoaderModelOnly": (0, "loras"),
    "CheckpointLoaderSimple": (0, "checkpoints"),
}


def wf_load(path=None):
    return json.load(open(path or CFG["workflow_json"], encoding="utf-8"))


def wf_save(wf, path=None):
    with open(path or CFG["workflow_json"], "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=1)


def wf_find(wf, ntype):
    return [n for n in wf["nodes"] if n.get("type") == ntype]


def wf_types(wf):
    return sorted({n.get("type") for n in wf["nodes"] if n.get("type")})


def wf_loaders(wf):
    """[(节点, widget下标, models子目录, 当前文件名), ...]"""
    out = []
    for n in wf["nodes"]:
        spec = LOADER_FIELDS.get(n.get("type"))
        if not spec:
            continue
        idx, sub = spec
        wv = n.get("widgets_values") or []
        if len(wv) > idx and isinstance(wv[idx], str):
            out.append((n, idx, sub, wv[idx]))
    return out


def director(wf):
    ns = wf_find(wf, "MiniMaxH3Director")
    return ns[0] if ns else None


def timeline_idx(node):
    """timeline_data 是唯一一个能解析成含 segments 的 JSON 字符串的 widget。"""
    for i, v in enumerate(node.get("widgets_values") or []):
        if isinstance(v, str) and v.strip().startswith("{") and "segments" in v:
            try:
                if isinstance(json.loads(v), dict):
                    return i
            except Exception:
                pass
    return -1


def timeline_get(node):
    i = timeline_idx(node)
    return (json.loads(node["widgets_values"][i]), i) if i >= 0 else (None, -1)


def timeline_set(node, tl, i):
    node["widgets_values"][i] = json.dumps(tl, ensure_ascii=False)


def widget_after(node, label, kind):
    """按「分组标题 + 类型」定位 widget，不写死下标（节点版本变了也不会错位）。"""
    wv = node.get("widgets_values") or []
    if label in wv:
        i = wv.index(label) + 1
        if i < len(wv) and isinstance(wv[i], kind) and not (isinstance(wv[i], bool) ^ (kind is bool)):
            return i
    return -1


def timeline_assets(tl):
    """时间线引用到的 input 素材文件名，按出现顺序去重。"""
    seen = []

    def add(v):
        if isinstance(v, str) and v and v not in seen:
            seen.append(v)

    for seg in tl.get("segments", []):
        add((seg.get("genImage") or {}).get("imageFile"))
        add((seg.get("endImage") or {}).get("imageFile"))
        for r in seg.get("refs") or []:
            add(r.get("imageFile") if isinstance(r, dict) else None)
    g = tl.get("global") or {}
    add((g.get("genImage") or {}).get("imageFile"))
    add((g.get("referenceVideo") or {}).get("videoFile"))
    for r in g.get("refs") or []:
        add(r.get("imageFile") if isinstance(r, dict) else None)
    for a in g.get("refAudios") or []:
        add(a.get("audioFile") if isinstance(a, dict) else None)
    add((tl.get("video") or {}).get("videoFile"))
    return seen



# ---------- 外网通道（Cell 8 用）----------
import shlex, urllib.request

def supported_flags():
    out = sh("%s main.py --help" % sys.executable, cwd=COMFY, check=False, quiet=True)
    return set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9_.-]*", out))


def sanitize(extra, known):
    if not known:
        return extra, []
    toks, keep, dropped, i = shlex.split(extra), [], [], 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            good = t.split("=", 1)[0] in known
            (keep if good else dropped).append(t)
            i += 1
            while i < len(toks) and not toks[i].startswith("-"):
                if good:
                    keep.append(toks[i])
                i += 1
        else:
            keep.append(t)
            i += 1
    return " ".join(keep), dropped


def start_frp():
    FRP, v = CFG["frp_dir"], CFG["frp_ver"]
    if not os.path.exists(FRP + "/frpc"):
        sh("wget -qO- https://github.com/fatedier/frp/releases/download/v%s/"
           "frp_%s_linux_amd64.tar.gz | tar -xz -C %s" % (v, v, ROOT), check=False)
    if not os.path.exists(FRP + "/frpc"):
        log("frpc 没下下来（GitHub 出口不通）", "!")
        return None
    sh("chmod +x %s/frpc" % FRP, quiet=True)

    conf = ['serverAddr = "%s"' % CFG["frp_host"],
            "serverPort = %d" % CFG["frp_port"],
            "loginFailExit = false",
            "transport.tcpMux = true",
            "transport.poolCount = 5",
            "transport.heartbeatInterval = 15",
            "transport.heartbeatTimeout = 60",
            'log.to = "%s/frpc.log"' % ROOT,
            'log.level = "info"']
    if CFG["frp_token"]:
        conf.insert(3, 'auth.token = "%s"' % CFG["frp_token"])
    conf += ["", "[[proxies]]",
             'name = "comfyui_colab_%d"' % CFG["remote_port"],
             'type = "tcp"',
             'localIP = "127.0.0.1"',
             "localPort = %d" % PORT,
             "remotePort = %d" % CFG["remote_port"]]
    open(FRP + "/frpc.toml", "w").write("\n".join(conf) + "\n")
    log("frpc %s -> %s:%d，远端端口 %d"
        % (v, CFG["frp_host"], CFG["frp_port"], CFG["remote_port"]))

    subprocess.Popen("%s/frpc -c %s/frpc.toml >> %s/frpc.log 2>&1"
                     % (FRP, FRP, ROOT), shell=True)

    #   frps 侧常见拒绝原因，直接翻译成人话，不用去翻日志
    BAD = [("port not allowed",
            "frps 的 allowPorts 没放行 %d，在 frps.toml 里放行或把 remote_port 改成已放行的端口"
            % CFG["remote_port"]),
           ("port already used",
            "远端 %d 被占（上一个会话的 frpc 还挂着），等 30 秒重试或换 remote_port"
            % CFG["remote_port"]),
           ("token in login doesn't match", "frp_token 和 frps 对不上"),
           ("authorization failed", "frp_token 和 frps 对不上"),
           ("login to server failed",
            "连不上 %s:%d（frps 没跑 / 端口没开 / 域名解析不对）"
            % (CFG["frp_host"], CFG["frp_port"]))]
    txt = ""
    for _ in range(24):
        time.sleep(1.5)
        if os.path.exists(ROOT + "/frpc.log"):
            txt = open(ROOT + "/frpc.log", errors="ignore").read()
        if "start proxy success" in txt:
            return "http://%s:%d" % (CFG["frp_host"], CFG["remote_port"])
        for key, why in BAD:
            if key in txt:
                log("frp 失败：" + why, "!")
                return None
    log("frp 等了 36 秒没出 start proxy success，日志尾部：\n" + txt[-800:], "!")
    return None


def start_colab_proxy():
    if not IN_COLAB:
        return None
    from google.colab import output
    output.serve_kernel_port_as_window(PORT)
    return "Colab 端口代理新窗口（弹窗被拦就允许一下）"


# ---------- 模型落盘（Cell 5 用）/ ComfyUI 接口（Cell 8 用）----------
def link_real(src, final):
    """HF 缓存里是相对符号链接，先 realpath 解成真 blob 再硬链，
    否则换目录后立刻悬空 -> getsize 报 [Errno 2]。"""
    real = os.path.realpath(src)
    if not os.path.isfile(real):
        raise RuntimeError("HF 缓存解析失败: %s -> %s" % (src, real))
    if os.path.lexists(final) and not os.path.exists(final):
        os.unlink(final)                       # 清掉上一轮的悬空链接
    if os.path.lexists(final):
        return real
    try:
        os.link(real, final)
    except OSError:
        try:
            os.symlink(real, final)
        except OSError:
            shutil.copy2(real, final)
    return real


def safetensors_ok(path):
    size = os.path.getsize(os.path.realpath(path))
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n + 8 > size:
            return False, size
        json.loads(f.read(n))
    return True, size


def add_diffusion_prefix(src, dst, prefix="diffusion_model."):
    """作者的 LoRA 键是 blocks.0.*，ComfyUI 要 diffusion_model. 命名空间。
    只重写 JSON 头，数据区原样拷贝，不用 import torch。"""
    tmp = dst + ".part"
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        meta = head.pop("__metadata__", None)
        new, renamed = {}, 0
        for k, v in head.items():
            if k.startswith(prefix):
                new[k] = v
            else:
                new[prefix + k] = v
                renamed += 1
        if meta is not None:
            new = dict([("__metadata__", meta)] + list(new.items()))
        hb = json.dumps(new, separators=(",", ":")).encode("utf-8")
        hb += b" " * ((-len(hb)) % 8)
        with open(tmp, "wb") as o:
            o.write(struct.pack("<Q", len(hb)))
            o.write(hb)
            shutil.copyfileobj(f, o, 8 * 1024 ** 2)
    os.replace(tmp, dst)
    log("LoRA 键名已加前缀：%d 个 tensor" % renamed)


def fetch(repo, filename, subdir, rename):
    dest_dir = os.path.join(COMFY, "models", subdir)
    os.makedirs(dest_dir, exist_ok=True)
    base = rename or os.path.basename(filename)
    final = os.path.join(dest_dir, base)
    if os.path.exists(final) and os.path.getsize(os.path.realpath(final)) > 100 * 1024 ** 2:
        return base, "跳过(已存在) %.1f GB" % (os.path.getsize(os.path.realpath(final)) / 1024 ** 3)
    src = hf_hub_download(repo_id=repo, filename=filename, repo_type="model")
    if rename and repo == CFG["lora_repo"]:
        add_diffusion_prefix(os.path.realpath(src), final)
    else:
        link_real(src, final)
    ok, size = safetensors_ok(final)
    if not ok:
        raise RuntimeError("%s 头部校验失败，文件不完整" % base)
    return base, "完成 %.1f GB" % (size / 1024 ** 3)


def api(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=60) as r:
        return json.load(r)
