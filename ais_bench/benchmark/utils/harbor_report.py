#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 AISBench/Harbor agent 评测 run 生成自包含的 index.html 报告。

由 DeepSWESummarizer.summarize() 尾部调用：
    from ais_bench.benchmark.utils.harbor_report import generate_html_report
    generate_html_report(work_dir, logger=self.logger)

trajectory 解析基于 ATIF（Agent Trajectory Interchange Format）标准 schema，
与具体 agent 无关：message / observation.content 均兼容 str 与
ContentPart[]（多模态），tool_calls 通用渲染（优先高亮 command 类字段）。
不做任何截断。

<run_dir> 目录结构（HarborRunner 产物）:
    results/<agent>/<dataset>.json                 汇总分数
    results/<agent>/<dataset>/details/<trial>/     单 trial 产物
        result.json / agent/trajectory.json / verifier/{ctrf,reward}.json
        artifacts/logs/artifacts/model.patch
    summary/summary_*.csv

输出: <run_dir>/index.html —— 所有数据内嵌，file:// 直接打开，离线可用。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _dur(a, b):
    ta, tb = _ts(a), _ts(b)
    if ta and tb:
        return round((tb - ta).total_seconds(), 1)
    return None


def _atif_text(content) -> str:
    """ATIF 通用文本提取：兼容 str 与 ContentPart[]（ATIF-v1.6 多模态）。

    图片部分显示为占位符（报告为纯文本渲染，不内嵌图片二进制）。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text") or "")
                elif p.get("type") == "image":
                    src = p.get("source") or {}
                    parts.append(f'[image: {src.get("path", "?")}]')
                else:
                    parts.append(json.dumps(p, ensure_ascii=False))
            else:
                parts.append(str(p))
        return "\n".join(x for x in parts if x)
    return str(content)


def _slim_ctrf(ctrf: dict) -> dict:
    """ctrf 只保留 summary 和精简后的 tests，控制报告体积。"""
    results = ctrf.get("results") or {}
    tests = []
    for t in results.get("tests") or []:
        item = {"name": t.get("name", ""), "status": t.get("status", "")}
        if t.get("message"):
            item["message"] = t["message"]
        if t.get("duration"):
            item["duration"] = t["duration"]
        tests.append(item)
    return {"summary": results.get("summary") or {}, "tests": tests}


_CMD_CLIP = 120
_OUT_CLIP = 200


def _clip(s, n):
    """截断字符串到 n 字符，超长加省略号（用于摘要，控制报告体积）。"""
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + "…"


def _tool_sample(tool: dict) -> str | None:
    """提取工具调用的一个典型命令（优先 command/cmd/code）。"""
    args = tool.get("args") or {}
    for k in ("command", "cmd"):
        if isinstance(args.get(k), str) and args[k]:
            return args[k]
    if isinstance(args.get("code"), str) and args["code"]:
        return args["code"]
    return None


def _aggregate_tool_stats(steps: list) -> list:
    """按工具聚合调用次数与 step 级耗时（估算）。

    耗时口径：ATIF 未记录单工具 wall-clock；用相邻 step.timestamp 差值估算，
    同一 step 内并行调用的多个工具共享该 step 耗时（不可拆分）。
    """
    agg: dict = {}
    for s in steps:
        d = s.get("dur")
        valid_dur = d if (d is not None and d >= 0) else None
        tools = s.get("tools") or []
        seen = set()
        for t in tools:
            fn = t.get("fn") or "unknown"
            st = agg.setdefault(
                fn,
                {"fn": fn, "count": 0, "step_durs": [], "parallel": 0, "sample": None},
            )
            st["count"] += 1
            if fn not in seen:
                seen.add(fn)
                if valid_dur is not None:
                    st["step_durs"].append(valid_dur)
            if len(tools) > 1:
                st["parallel"] += 1
            if st["sample"] is None:
                st["sample"] = _tool_sample(t)
    out = []
    for st in agg.values():
        durs = sorted(st["step_durs"])
        if durs:
            m = len(durs) // 2
            median = durs[m] if len(durs) % 2 else (durs[m - 1] + durs[m]) / 2
        else:
            median = None
        out.append({
            "fn": st["fn"],
            "count": st["count"],
            "n_steps": len(durs),
            "total_dur": round(sum(durs), 1) if durs else None,
            "avg_dur": round(sum(durs) / len(durs), 1) if durs else None,
            "median_dur": round(median, 1) if median is not None else None,
            "max_dur": round(max(durs), 1) if durs else None,
            "parallel": st["parallel"],
            "sample": st["sample"],
        })
    out.sort(key=lambda x: (-x["count"], x["fn"]))
    return out


def _slim_trajectory(traj: dict) -> dict:
    """ATIF 轨迹通用解析（不截断）。

    只依赖 ATIF schema 字段，任何 agent 输出的标准 trajectory.json 均可解析：
    - message / observation.content: str | ContentPart[]
    - tool_calls: [{function_name, arguments}]，arguments 原样保留
    - reasoning_content: agent 显式思考链
    - observation.results[].source_call_id: 把每个工具输出精确关联回 tool_call
    """
    steps = []
    flat_calls = []
    prev_ts = None
    for s in traj.get("steps") or []:
        tools = []
        for tc in s.get("tool_calls") or []:
            tools.append({
                "id": tc.get("tool_call_id", ""),
                "fn": tc.get("function_name", ""),
                "args": tc.get("arguments") or {},
                "out": "",
            })
        ob = s.get("observation") or {}
        results = ob.get("results") or []
        obs_parts = []
        for r in results:
            txt = _atif_text(r.get("content"))
            if txt:
                obs_parts.append(txt)
        obs = "\n".join(obs_parts)
        # 用 source_call_id 把结果一一对应回工具（并行多工具也能区分输出归属）
        by_id = {t["id"]: t for t in tools if t["id"]}
        for r in results:
            cid = r.get("source_call_id")
            if cid and cid in by_id:
                by_id[cid]["out"] = _atif_text(r.get("content"))
        m = s.get("metrics") or {}
        ts = s.get("timestamp")
        # step 级 wall-clock 估算：相邻 step 时间戳差
        dur = _dur(prev_ts, ts)
        prev_ts = ts
        step = {
            "i": s.get("step_id"),
            "ts": ts,
            "dur": dur,
            "src": s.get("source"),
            "msg": _atif_text(s.get("message")),
            "reasoning": s.get("reasoning_content") or "",
            "tools": tools,
            "tool_names": [t["fn"] for t in tools if t["fn"]],
            "obs": obs,
            "ptok": m.get("prompt_tokens"),
            "ctok": m.get("completion_tokens"),
        }
        for t in tools:
            flat_calls.append({
                "step": s.get("step_id"),
                "ts": ts,
                "dur": dur,
                "fn": t.get("fn") or "unknown",
                "id": t.get("id", ""),
                "cmd": _clip(_tool_sample(t), _CMD_CLIP),
                "out": _clip(t.get("out"), _OUT_CLIP),
            })
        steps.append(step)
    return {
        "n_steps": len(steps),
        "steps": steps,
        "flat_calls": flat_calls,
        "tool_stats": _aggregate_tool_stats(steps),
    }


def _badcase_notes(trial_result: dict, reward: dict, failed: list) -> list:
    """自动归因（分类级，非语义级），返回结构化条目供前端排版。

    规则：
    ① exception_info 非空 → 执行异常（环境/依赖/网络问题，非模型问题）
    ② 失败用例名以 [f2p] 开头 → 新功能实现不完全（模型能力问题）
    ③ 失败用例名以 [p2p] 开头 → 回归（改动破坏已有功能，性质更严重）
    """
    notes = []
    exc = trial_result.get("exception_info")
    if exc:
        notes.append({
            "type": "exception",
            "title": "执行异常",
            "detail": f"{exc.get('exception_type', 'unknown')} — "
                      f"{str(exc.get('exception_message', ''))[:200]}",
            "cases": [],
        })
    f2p_fail = [t for t in failed if t["name"].startswith("[f2p]")]
    p2p_fail = [t for t in failed if t["name"].startswith("[p2p]")]
    if f2p_fail:
        notes.append({
            "type": "f2p",
            "title": "F2P（新功能）用例失败",
            "detail": f"{len(f2p_fail)} 个：agent 补丁未完整实现任务要求的行为。",
            "cases": [t["name"] for t in f2p_fail],
        })
    if p2p_fail:
        notes.append({
            "type": "p2p",
            "title": "P2P（回归）用例失败",
            "detail": f"{len(p2p_fail)} 个：agent 改动破坏了已有功能。",
            "cases": [t["name"] for t in p2p_fail],
        })
    if not notes and reward:
        notes.append({"type": "ok", "title": "未发现失败用例与执行异常", "detail": "", "cases": []})
    return notes


def _collect_trial(tdir: Path) -> dict:
    trial: dict = {"name": tdir.name}
    result = None
    p = tdir / "result.json"
    if p.exists():
        result = _load(p)
        trial["result"] = result
    p = tdir / "verifier/reward.json"
    if p.exists():
        trial["reward"] = _load(p)
    p = tdir / "verifier/ctrf.json"
    if p.exists():
        trial["ctrf"] = _slim_ctrf(_load(p))
    # ATIF trajectory：优先 harbor 归一化副本，回退 agent 原始输出（*.trajectory.json）
    traj_path = tdir / "agent/trajectory.json"
    if not traj_path.exists():
        cands = sorted((tdir / "agent").glob("*.trajectory.json")) if (tdir / "agent").is_dir() else []
        traj_path = cands[0] if cands else traj_path
    if traj_path.exists():
        trial["trajectory"] = _slim_trajectory(_load(traj_path))
    p = tdir / "artifacts/logs/artifacts/model.patch"
    if p.exists():
        trial["patch"] = p.read_text(encoding="utf-8", errors="replace")

    # 派生字段
    r = result or {}
    ar = r.get("agent_result") or {}
    trial["task_name"] = r.get("task_name", "")
    trial["model"] = (r.get("agent_info") or {}).get("model_info", {}).get("name", "")
    trial["agent_ver"] = (r.get("agent_info") or {}).get("version", "")
    trial["tokens"] = {
        "input": ar.get("n_input_tokens"),
        "cache": ar.get("n_cache_tokens"),
        "output": ar.get("n_output_tokens"),
    }
    trial["timings"] = {
        "total": _dur(r.get("started_at"), r.get("finished_at")),
        "env_setup": _dur(
            (r.get("environment_setup") or {}).get("started_at"),
            (r.get("environment_setup") or {}).get("finished_at"),
        ),
        "agent_setup": _dur(
            (r.get("agent_setup") or {}).get("started_at"),
            (r.get("agent_setup") or {}).get("finished_at"),
        ),
        "agent_exec": _dur(
            (r.get("agent_execution") or {}).get("started_at"),
            (r.get("agent_execution") or {}).get("finished_at"),
        ),
        "verifier": _dur(
            (r.get("verifier") or {}).get("started_at"),
            (r.get("verifier") or {}).get("finished_at"),
        ),
    }
    rw = trial.get("reward") or {}
    trial["resolved"] = rw.get("reward", 0) >= 1
    failed = [t for t in (trial.get("ctrf", {}).get("tests") or [])
              if t.get("status") != "passed"]
    trial["failed_tests"] = failed
    trial["badcase_notes"] = _badcase_notes(r, rw, failed)
    return trial


def _collect_run(run_dir: Path) -> dict:
    data: dict = {"run_dir": run_dir.name, "agents": [], "summary_csv": None}

    summary_dir = run_dir / "summary"
    if summary_dir.is_dir():
        for csv_file in sorted(summary_dir.glob("summary_*.csv")):
            data["summary_csv"] = csv_file.read_text(encoding="utf-8", errors="replace")
            break

    results_root = run_dir / "results"
    if not results_root.is_dir():
        raise FileNotFoundError(f"未找到 results 目录: {results_root}")

    for agg_file in sorted(results_root.glob("*/*.json")):
        agent_name, dataset = agg_file.parent.name, agg_file.stem
        details = agg_file.parent / dataset / "details"
        trials = []
        if details.is_dir():
            for tdir in sorted(p for p in details.iterdir() if p.is_dir()):
                trials.append(_collect_trial(tdir))
        data["agents"].append(
            {
                "agent": agent_name,
                "dataset": dataset,
                "aggregate": _load(agg_file),
                "trials": trials,
            }
        )

    if not data["agents"]:
        raise FileNotFoundError(f"未在 {results_root} 下找到 <agent>/<dataset>.json")
    return data


TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harbor 评测报告</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#1c2330;--sub:#66707f;--line:#e3e6ec;--acc:#2563eb;--ok:#16a34a;--bad:#dc2626;--chip:#eef2f7}
*{box-sizing:border-box}
body{margin:0;font:14px/1.6 -apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;color:var(--ink);background:var(--bg)}
header{background:#111827;color:#fff;padding:14px 22px;position:sticky;top:0;z-index:20}
header h1{margin:0;font-size:17px;font-weight:600}
header .sub{color:#9aa4b2;font-size:12px;margin-top:2px}
.wrap{display:flex;min-height:calc(100vh - 62px)}
nav{width:250px;background:var(--card);border-right:1px solid var(--line);padding:14px;flex-shrink:0;position:sticky;top:62px;align-self:flex-start;max-height:calc(100vh - 62px);overflow-y:auto}
nav .grp{font-size:11px;color:var(--sub);text-transform:uppercase;letter-spacing:.05em;margin:12px 0 6px}
nav .item{display:block;width:100%;text-align:left;padding:7px 10px;border:0;background:none;border-radius:6px;cursor:pointer;font-size:13px;color:var(--ink);font-family:inherit}
nav .item:hover{background:var(--chip)}
nav .item.on{background:var(--acc);color:#fff}
nav .trial{font-size:12px;padding:6px 10px;word-break:break-all}
main{flex:1;padding:20px 24px;min-width:0}
.tabs{display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap}
.tab{padding:7px 16px;border-radius:999px;border:1px solid var(--line);background:var(--card);cursor:pointer;font-size:13px;font-family:inherit}
.tab.on{background:#111827;color:#fff;border-color:#111827}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:16px}
.card h3{margin:0 0 12px;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .k{font-size:12px;color:var(--sub)}
.kpi .v{font-size:21px;font-weight:700;margin-top:2px;font-variant-numeric:tabular-nums;word-break:break-all}
.kpi .v.good{color:var(--ok)}.kpi .v.badc{color:var(--bad)}
.kpi .n{font-size:11px;color:var(--sub);margin-top:2px}
.bar{height:8px;background:var(--chip);border-radius:4px;overflow:hidden;margin-top:8px}
.bar i{display:block;height:100%;background:var(--ok)}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:600}
.b-ok{background:#dcfce7;color:#15803d}.b-bad{background:#fee2e2;color:#b91c1c}.b-neu{background:var(--chip);color:var(--sub)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{font-size:12px;font-weight:600;color:var(--sub)}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
tr:hover td{background:#fafbfc}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.mono{font-family:ui-monospace,Consolas,Menlo,monospace;font-size:12px}
a{color:var(--acc)}
pre{background:#0f172a;color:#e2e8f0;border-radius:8px;padding:12px 14px;overflow:auto;font:12px/1.55 ui-monospace,Consolas,Menlo,monospace;max-height:420px;margin:8px 0 0}
.diff{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:0;overflow:auto;max-height:560px;font:12px/1.5 ui-monospace,Consolas,Menlo,monospace}
.diff div{padding:0 10px;white-space:pre}
.diff .add{background:#f0fdf4;color:#15803d}
.diff .del{background:#fef2f2;color:#b91c1c}
.diff .hunk{background:var(--chip);color:var(--sub)}
details.step{border:1px solid var(--line);border-radius:8px;margin-bottom:8px;background:var(--card);overflow:hidden}
details.step summary{cursor:pointer;padding:8px 12px;font-size:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
details.step summary:hover{background:var(--chip)}
details.step .body{padding:0 12px 12px;border-top:1px solid var(--line)}
.step .who{font-weight:700;font-size:11px;padding:1px 7px;border-radius:999px;background:var(--chip);color:var(--sub)}
.step .who.agent{background:#dbeafe;color:#1d4ed8}
.step .who.user{background:#fef3c7;color:#b45309}
.step .who.system{background:var(--chip);color:var(--sub)}
.step .cmd{background:#0f172a;color:#a5f3fc;padding:8px 12px;border-radius:6px;margin:6px 0;white-space:pre-wrap;word-break:break-all}
.step .fn{background:#0f172a;color:#c4b5fd;padding:8px 12px;border-radius:6px;margin:6px 0;white-space:pre-wrap;word-break:break-all}
.step .lab{font-size:11px;color:var(--sub);margin-top:8px;text-transform:uppercase;letter-spacing:.04em}
.step .msg{white-space:pre-wrap;word-break:break-word;margin:4px 0 0;font-size:13px}
.filter{padding:7px 12px;border:1px solid var(--line);border-radius:8px;font:13px inherit;width:320px;max-width:100%}
.tokchart{display:flex;align-items:flex-end;gap:1px;height:60px;margin-top:10px}
.tokchart i{flex:1;min-width:2px;background:var(--acc);border-radius:1px 1px 0 0;opacity:.75}
.tokchart i.ctok{background:var(--ok)}
.legend{display:flex;gap:14px;font-size:11px;color:var(--sub);margin-top:6px;flex-wrap:wrap}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.timeline{display:flex;height:26px;border-radius:6px;overflow:hidden;margin-top:10px;font-size:11px;color:#fff}
.timeline div{display:flex;align-items:center;justify-content:center;white-space:nowrap;overflow:hidden}
.empty{color:var(--sub);padding:20px;text-align:center}
.note{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;font-size:12.5px;margin-bottom:16px}
ul.tight{margin:6px 0;padding-left:20px}
ul.tight li{margin:4px 0}
.bd-item{border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin-bottom:10px;background:#fafbfc}
.bd-head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.bd-detail{color:var(--sub);font-size:13px}
.bd-cases{margin-top:8px;border-top:1px dashed var(--line);padding-top:8px}
.bd-cases .mono{display:block;padding:2px 0;word-break:break-all}
@media (max-width:920px){.wrap{flex-direction:column}nav{width:100%;border-right:0;border-bottom:1px solid var(--line);max-height:260px;overflow:auto}main{padding:16px 12px}}
@media print{nav{display:none}main{max-width:none}}
</style>
</head>
<body>
<header>
  <h1>Harbor Agent 评测报告</h1>
  <div class="sub" id="hsub"></div>
</header>
<div class="wrap">
  <nav id="nav"></nav>
  <main id="main"></main>
</div>
<script>
"use strict";
const DATA = __DATA__;

const S = {run:0, agent:0, trial:0, tab:"overview", filter:"", toolFilter:"", trialFilter:"all"};
const esc = s => String(s??"").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = n => (n==null||isNaN(n)) ? "–" : Number(n).toLocaleString("en-US");
const pct = n => (n==null||isNaN(n)) ? "–" : (n*100).toFixed(1)+"%";
const dur = s => (s==null) ? "–" : (s>=60 ? Math.floor(s/60)+"m"+Math.round(s%60)+"s" : s.toFixed?s.toFixed(0)+"s" : s);
const badge = st => st==="passed" ? '<span class="badge b-ok">PASS</span>' : '<span class="badge b-bad">FAIL</span>';

function cur(){ return DATA.runs[S.run].agents[S.agent]; }
function curTrial(){ const a=cur(); return a.trials[S.trial]; }

function renderHeader(){
  const r = DATA.runs[S.run];
  document.getElementById("hsub").textContent =
    `run ${r.run_dir} · ${cur().agent} × ${cur().dataset} · ${new Date().toLocaleDateString("zh-CN")}`;
}

function renderNav(){
  let h = "";
  DATA.runs.forEach((r, ri) => {
    h += `<div class="grp">Run ${esc(r.run_dir)}</div>`;
    r.agents.forEach((ag, ai) => {
      h += `<button class="item ${ri===S.run&&ai===S.agent?"on":""}" onclick="selAgent(${ri},${ai})" style="font-weight:600">${esc(ag.agent)} / ${esc(ag.dataset)}</button>`;
      ag.trials.forEach((t, ti) => {
        h += `<button class="item trial ${ri===S.run&&ai===S.agent&&ti===S.trial?"on":""}" onclick="selTrial(${ri},${ai},${ti})">${t.resolved?'<span class="badge b-ok">✓</span>':'<span class="badge b-bad">✗</span>'} ${esc(t.name)}</button>`;
      });
    });
  });
  document.getElementById("nav").innerHTML = h;
}

/* 切换左侧栏条目时保持当前 tab 上下文：
   - 处于“总览”时点击 case → 跳到该 case 的“Trial 得分”；
   - 处于某个具体 tab（得分/用例/轨迹/Patch/Badcase）时点击其他 case →
     保持该 tab，切到新 case 的对应视图。 */
function selAgent(ri, ai){ S.run=ri; S.agent=ai; S.trial=0; if(S.tab==="overview") S.tab="trial"; renderAll(); }
function selTrial(ri, ai, ti){ S.run=ri; S.agent=ai; S.trial=ti; if(S.tab==="overview") S.tab="trial"; renderAll(); }
function selTab(t){ S.tab=t; renderMain(); }

function kpi(k, v, cls, note){ return `<div class="kpi"><div class="k">${esc(k)}</div><div class="v ${cls||""}">${v}</div>${note?`<div class="n">${esc(note)}</div>`:""}</div>`; }

/* ATIF 通用 tool_call 渲染：优先高亮 command 类字段，否则显示 fn + arguments */
function toolPreview(tc){
  const a = tc.args||{};
  for (const k of ["command","cmd"]) {
    if (typeof a[k] === "string" && a[k]) return "$ "+a[k];
  }
  if (typeof a.code === "string" && a.code) return a.code;
  return tc.fn + (Object.keys(a).length ? " "+JSON.stringify(a) : "");
}
function renderTools(tools){
  return (tools||[]).map(tc=>{
    const a = tc.args||{};
    for (const k of ["command","cmd"]) {
      if (typeof a[k] === "string" && a[k]) return `<div class="cmd">$ ${esc(a[k])}</div>`;
    }
    if (typeof a.code === "string" && a.code) return `<div class="cmd">${esc(a.code)}</div>`;
    return `<div class="fn">${esc(tc.fn)}${Object.keys(a).length? " "+esc(JSON.stringify(a)) : ""}</div>`;
  }).join("");
}

function aggOverview(){
  const ag = cur(), agg = ag.aggregate || {};
  const ts = ag.trials;
  const n = ts.length;
  const resolved = ts.filter(t=>t.resolved).length;
  const err = ts.filter(t=>t.result && t.result.exception_info).length;
  const avg = key => { const v=ts.map(t=>(t.reward||{})[key]).filter(x=>x!=null); return v.length? v.reduce((a,b)=>a+b,0)/v.length : null; };
  const tokIn = ts.reduce((a,t)=>a+((t.tokens||{}).input||0),0);
  const tokOut = ts.reduce((a,t)=>a+((t.tokens||{}).output||0),0);
  const durs = ts.map(t=>(t.timings||{}).total).filter(x=>x);
  const avgDur = durs.length? durs.reduce((a,b)=>a+b,0)/durs.length : null;

  let h = `<div class="card"><h3>总分概览 — ${esc(ag.agent)} × ${esc(ag.dataset)}</h3><div class="grid">
    ${kpi("Trials", n)}
    ${kpi("Resolved 率", pct(resolved/n), resolved===n?"good":(resolved?"":"badc"), `${resolved}/${n} 全部通过`)}
    ${kpi("avg F2P", pct(avg("f2p")), "", "fail-to-pass 新功能")}
    ${kpi("avg P2P", pct(avg("p2p")), "", "pass-to-pass 回归")}
    ${kpi("avg partial", pct(avg("partial")))}
    ${kpi("异常 trial", err, err?"badc":"good")}
    ${kpi("输入 tokens(含cache)", fmt(tokIn), "", "累加")}
    ${kpi("输出 tokens", fmt(tokOut))}
    ${kpi("平均耗时", dur(avgDur), "", "trial 全程")}
  </div></div>`;

  const scoreNote = (agg.avg_score!=null && agg.avg_score>1)
    ? `本 run 的汇总 JSON 由<b>旧版统计逻辑</b>生成（avg_score=${fmt(agg.avg_score)} 将 rewards 各字段误当分数求和，口径错误）；衡量效果请以 <b>F2P / P2P / partial / Resolved 率</b> 为准。`
    : `avg_score = 各 trial <code>reward</code> 字段均值（DeepSWE 下即 resolved 率）；F2P / P2P / partial 为分项指标。`;
  h += `<div class="note"><b>指标口径说明：</b>${scoreNote}</div>`;

  // reward distribution
  if ((agg.reward_distribution||[]).length){
    h += `<div class="card"><h3>reward 分布（汇总 JSON 原始口径）</h3><table><tr><th>score</th><th>count</th></tr>` +
      agg.reward_distribution.map(d=>`<tr><td>${fmt(d.score)}</td><td>${d.count}</td></tr>`).join("") + `</table></div>`;
  }

  // summary csv
  if (DATA.runs[S.run].summary_csv){
    const rows = DATA.runs[S.run].summary_csv.trim().split(/\r?\n/).map(l=>l.split(","));
    h += `<div class="card"><h3>summary CSV</h3><table><tr>${rows[0].map(c=>`<th>${esc(c)}</th>`).join("")}</tr>` +
      rows.slice(1).map(r=>`<tr>${r.map(c=>`<td>${esc(c)}</td>`).join("")}</tr>`).join("") + `</table></div>`;
  }

  // trial 表（可按状态筛选）
  const trFilter = S.trialFilter || "all";
  const filtered = ts.map((t,i)=>({t,i})).filter(({t})=>{
    if (trFilter==="resolved") return t.resolved;
    if (trFilter==="unresolved") return !t.resolved;
    if (trFilter==="exception") return !!(t.result && t.result.exception_info);
    return true;
  });
  h += `<div class="card"><h3>Trials（${filtered.length} / ${n}）</h3>
    <div style="margin-bottom:10px">状态筛选：
      <select class="filter" onchange="S.trialFilter=this.value;renderMain()">
        <option value="all"${trFilter==="all"?" selected":""}>全部</option>
        <option value="resolved"${trFilter==="resolved"?" selected":""}>Resolved</option>
        <option value="unresolved"${trFilter==="unresolved"?" selected":""}>Unresolved</option>
        <option value="exception"${trFilter==="exception"?" selected":""}>异常</option>
      </select>
    </div><table>
    <tr><th>trial</th><th>状态</th><th class="num">F2P</th><th class="num">P2P</th><th class="num">partial</th><th class="num">输入tok</th><th class="num">输出tok</th><th class="num">耗时</th><th class="num">失败用例</th></tr>` +
    filtered.map(({t,i})=>{
      const rw = t.reward||{};
      return `<tr style="cursor:pointer" onclick="selTrial(${S.run},${S.agent},${i})">
        <td class="mono">${esc(t.name)}</td>
        <td>${t.resolved?'<span class="badge b-ok">RESOLVED</span>':'<span class="badge b-bad">UNRESOLVED</span>'}</td>
        <td class="num">${rw.f2p!=null? rw.f2p_passed+"/"+rw.f2p_total : "–"}</td>
        <td class="num">${rw.p2p!=null? rw.p2p_passed+"/"+rw.p2p_total : "–"}</td>
        <td class="num">${pct(rw.partial)}</td>
        <td class="num">${fmt((t.tokens||{}).input)}</td>
        <td class="num">${fmt((t.tokens||{}).output)}</td>
        <td class="num">${dur((t.timings||{}).total)}</td>
        <td class="num">${(t.failed_tests||[]).length}</td>
      </tr>`;
    }).join("") + `</table></div>`;
  return h;
}

function trialOverview(){
  const t = curTrial(), rw = t.reward||{}, tk = t.tokens||{}, tm = t.timings||{};
  const bar = (v)=>`<div class="bar"><i style="width:${Math.max(0,Math.min(100,(v||0)*100))}%"></i></div>`;
  let h = `<div class="card"><h3>${esc(t.name)} — 得分</h3><div class="grid">
      ${kpi("Resolved", t.resolved?"YES":"NO", t.resolved?"good":"badc", "reward≥1 判定")}
      ${kpi("F2P", (rw.f2p_passed??"–")+"/"+(rw.f2p_total??"–"), rw.f2p>=1?"good":(rw.f2p!=null?"badc":""), pct(rw.f2p))}
      ${kpi("P2P", (rw.p2p_passed??"–")+"/"+(rw.p2p_total??"–"), rw.p2p>=1?"good":(rw.p2p!=null?"badc":""), pct(rw.p2p))}
      ${kpi("partial", pct(rw.partial))}
      ${kpi("输入 tokens", fmt(tk.input), "", "其中 cache "+fmt(tk.cache))}
      ${kpi("输出 tokens", fmt(tk.output))}
      ${kpi("轨迹步数", (t.trajectory||{}).n_steps??"–")}
    </div>
    <div class="grid" style="margin-top:12px">
      <div class="kpi"><div class="k">F2P 通过率</div><div class="v">${pct(rw.f2p)}</div>${bar(rw.f2p)}</div>
      <div class="kpi"><div class="k">P2P 通过率</div><div class="v">${pct(rw.p2p)}</div>${bar(rw.p2p)}</div>
      <div class="kpi"><div class="k">partial 得分</div><div class="v">${pct(rw.partial)}</div>${bar(rw.partial)}</div>
    </div></div>`;

  // 时间线
  const segs = [["env_setup",tm.env_setup,"#94a3b8"],["agent_setup",tm.agent_setup,"#60a5fa"],["agent_exec",tm.agent_exec,"#2563eb"],["verifier",tm.verifier,"#16a34a"]];
  const tot = segs.map(s=>s[1]||0).reduce((a,b)=>a+b,0) || 1;
  h += `<div class="card"><h3>耗时（总 ${dur(tm.total)}）</h3><div class="timeline">` +
    segs.map(([n,v,c])=>`<div title="${n}: ${dur(v)}" style="width:${(v||0)/tot*100}%;background:${c}">${(v||0)/tot>0.12? esc(n)+" "+dur(v):""}</div>`).join("") +
    `</div><div class="legend">` + segs.map(([n,v,c])=>`<span><i style="background:${c}"></i>${esc(n)} ${dur(v)}</span>`).join("") + `</div></div>`;

  h += `<div class="card"><h3>执行信息</h3><table>
    <tr><th>task</th><td class="mono">${esc(t.task_name)}</td><th>model</th><td class="mono">${esc(t.model)} (agent v${esc(t.agent_ver)})</td></tr>
    <tr><th>异常</th><td colspan="3">${t.result&&t.result.exception_info? '<span class="badge b-bad">'+esc(t.result.exception_info.exception_type||"exception")+'</span>' : '<span class="badge b-ok">无</span>'}</td></tr>
  </table></div>`;
  return h;
}

function trialTests(){
  const t = curTrial(), c = t.ctrf;
  if (!c) return `<div class="empty">该 trial 无 CTRF 数据</div>`;
  const sum = c.summary||{};
  const tests = c.tests||[];
  const shown = tests.filter(x=>!S.filter || x.name.toLowerCase().includes(S.filter.toLowerCase()) || x.status!=="passed");
  let h = `<div class="card"><h3>测试用例（CTRF）</h3><div class="grid">
    ${kpi("总数", fmt(sum.tests))}
    ${kpi("通过", fmt(sum.passed), "good")}
    ${kpi("失败", fmt(sum.failed), sum.failed?"badc":"good")}
    ${kpi("跳过", fmt(sum.skipped))}
  </div>
  <div style="margin-top:12px"><input class="filter" placeholder="按名称过滤（默认显示全部失败用例 + 匹配项）" value="${esc(S.filter)}" oninput="S.filter=this.value;renderMain()"></div>
  <div style="max-height:560px;overflow:auto;margin-top:10px"><table>
    <tr><th style="width:70px">状态</th><th>用例</th></tr>` +
    shown.map(x=>{
      const f = x.status!=="passed";
      return `<tr><td>${badge(x.status)}</td><td class="mono">${esc(x.name)}${f&&x.message?`<pre>${esc(x.message)}</pre>`:""}</td></tr>`;
    }).join("") +
    `</table><div class="n" style="padding:8px;color:#66707f;font-size:12px">显示 ${shown.length} / ${tests.length}</div></div></div>`;
  return h;
}

function trialTrace(){
  const t = curTrial(), tr = t.trajectory;
  if (!tr || !tr.steps) return `<div class="empty">该 trial 无轨迹数据</div>`;
  const steps = tr.steps;
  const maxTok = Math.max(...steps.map(s=>s.ptok||0), 1);
  // token 曲线（主区域 prompt tokens）
  let h = `<div class="card"><h3>Agent 执行轨迹（${steps.length} 步）</h3>
    <div class="tokchart">` + steps.map(s=>`<i title="step ${s.i}: prompt ${fmt(s.ptok)} / completion ${fmt(s.ctok)}" style="height:${Math.max(2,(s.ptok||0)/maxTok*100)}%"></i>`).join("") + `</div>
    <div class="legend"><span><i style="background:var(--acc)"></i>每步 prompt tokens（悬停查看明细）</span></div></div>`;
  h += steps.map(s=>{
    const tools = renderTools(s.tools);
    const obs = s.obs? `<div class="lab">observation</div><pre>${esc(s.obs)}</pre>` : "";
    const reasoning = s.reasoning? `<div class="lab">reasoning</div><div class="msg">${esc(s.reasoning)}</div>` : "";
    const msg = s.msg? `<div class="lab">message</div><div class="msg">${esc(s.msg)}</div>` : "";
    const t0 = (s.tools||[])[0];
    const head = `<span class="who ${esc(s.src)}">${esc(s.src)}</span> <b>#${s.i}</b> <span style="color:var(--sub)">${esc((s.ts||"").slice(11,19))}</span>` +
      (t0? ` <span class="mono" style="color:#0369a1">${esc(toolPreview(t0).slice(0,90))}${toolPreview(t0).length>90?"…":""}</span>` : "");
    return `<details class="step"><summary>${head}</summary><div class="body">${reasoning}${msg}${tools}${obs}</div></details>`;
  }).join("");
  return h;
}

function trialTools(){
  const t = curTrial(), tr = t.trajectory;
  if (!tr || !tr.steps) return `<div class="empty">该 trial 无轨迹数据</div>`;
  const flat = tr.flat_calls || [];
  const stats = tr.tool_stats || [];
  const PALETTE = ["#2563eb","#16a34a","#ca8a04","#dc2626","#7c3aed","#0891b2","#db2777","#65a30d"];
  const colorOf = {};
  stats.forEach((s,i)=>colorOf[s.fn]=PALETTE[i%PALETTE.length]);
  const dd = s => (s==null) ? "–" : (s>=60 ? Math.floor(s/60)+"m"+Math.round(s%60)+"s" : (Math.round(s*10)/10)+"s");

  const durs = flat.map(c=>c.dur).filter(x=>x!=null&&x>=0);
  const totalDur = durs.reduce((a,b)=>a+b,0);
  const avgDur = durs.length? totalDur/durs.length : null;
  const maxDur = durs.length? Math.max(...durs) : null;
  const top = stats[0];

  let h = `<div class="card"><h3>工具调用统计</h3><div class="grid">
    ${kpi("工具调用总次数", flat.length)}
    ${kpi("工具种类", stats.length)}
    ${kpi("最常用工具", top? esc(top.fn)+" ×"+top.count : "–")}
    ${kpi("step 总耗时", dd(totalDur), "", "相邻 timestamp 差值累加")}
    ${kpi("平均 step 耗时", dd(avgDur), "", "估算口径")}
    ${kpi("最大 step 耗时", dd(maxDur))}
  </div>
  <div class="note" style="margin-top:12px"><b>耗时口径：</b>ATIF 未记录单工具 wall-clock，此处耗时=相邻 <code>step.timestamp</code> 差值（=工具执行 + 下次 LLM 推理的复合时间，估算值）。同一 step 内并行的多工具共享该耗时。</div></div>`;

  if (stats.length){
    h += `<div class="card"><h3>工具聚合</h3><table>
      <tr><th>工具</th><th class="num">调用次数</th><th class="num">平均耗时</th><th class="num">中位耗时</th><th class="num">最大耗时</th><th class="num">累计耗时</th><th class="num">并行次数</th></tr>` +
      stats.map(s=>`<tr>
        <td class="mono">${esc(s.fn)}</td>
        <td class="num">${fmt(s.count)}</td>
        <td class="num">${dd(s.avg_dur)}</td>
        <td class="num">${dd(s.median_dur)}</td>
        <td class="num">${dd(s.max_dur)}</td>
        <td class="num">${dd(s.total_dur)}</td>
        <td class="num">${fmt(s.parallel)}</td>
      </tr>`).join("") + `</table></div>`;
  }

  if (flat.length){
    const tot = totalDur || 1;
    h += `<div class="card"><h3>工具调用时间线（每个色块 = 一次调用）</h3>
      <div class="timeline">` + flat.map(c=>`<div title="#${c.step} ${esc(c.fn)} ${dd(c.dur)} — ${esc(c.cmd)}" style="width:${((c.dur||0)/tot*100).toFixed(2)}%;background:${colorOf[c.fn]||"#94a3b8"}"></div>`).join("") + `</div>
      <div class="legend">` + stats.map(s=>`<span><i style="background:${colorOf[s.fn]}"></i>${esc(s.fn)}</span>`).join("") + `</div></div>`;

    const shown = flat.filter(c=>!S.toolFilter || (c.fn||"").toLowerCase().includes(S.toolFilter.toLowerCase()) || (c.cmd||"").toLowerCase().includes(S.toolFilter.toLowerCase()));
    h += `<div class="card"><h3>逐次调用明细（${shown.length} / ${flat.length}）</h3>
      <div style="margin-bottom:12px"><input class="filter" placeholder="按工具名或命令过滤" value="${esc(S.toolFilter||"")}" oninput="S.toolFilter=this.value;renderMain()"></div>
      <div style="max-height:560px;overflow:auto"><table>
        <tr><th class="num">step</th><th>工具</th><th>命令（截断 120）</th><th class="num">耗时</th><th>输出摘要（截断 200）</th></tr>` +
      shown.map(c=>`<tr>
        <td class="num">#${c.step}</td>
        <td class="mono">${esc(c.fn)}</td>
        <td class="mono">${esc(c.cmd)}</td>
        <td class="num">${dd(c.dur)}</td>
        <td class="mono">${esc(c.out)}</td>
      </tr>`).join("") + `</table></div></div>`;
  }
  return h;
}

function trialPatch(){
  const t = curTrial();
  if (!t.patch) return `<div class="empty">该 trial 无 model.patch</div>`;
  const lines = t.patch.split("\n");
  const body = lines.map(l=>{
    const cls = l.startsWith("+")?"add":(l.startsWith("-")?"del":(l.startsWith("@@")?"hunk":""));
    return `<div class="${cls}">${esc(l)}</div>`;
  }).join("");
  return `<div class="card"><h3>model.patch（agent 产物 diff，${lines.length} 行）</h3><div class="diff">${body}</div></div>`;
}

function trialBadcase(){
  const t = curTrial();
  let h = `<div class="card"><h3>Badcase 分析 — ${esc(t.name)}</h3>
    <div class="note" style="margin-bottom:14px">归因规则：①执行异常=环境/依赖问题（非模型）②F2P 失败=新功能实现不完全（模型）③P2P 失败=回归（模型，破坏已有功能）</div>`;
  if (t.resolved && !(t.failed_tests||[]).length){
    h += `<p class="badge b-ok" style="font-size:13px;padding:4px 12px">RESOLVED — 无 badcase</p></div>`;
    return h;
  }
  const notes = t.badcase_notes||[];
  if (notes.length){
    h += notes.map(n=>{
      if (n.type==="ok") return `<p class="badge b-ok" style="font-size:13px;padding:4px 12px">${esc(n.title)}</p>`;
      const cls = n.type==="exception" ? "b-neu" : "b-bad";
      let body = `<div class="bd-head"><span class="badge ${cls}">${esc(n.title)}</span><span class="bd-detail">${esc(n.detail)}</span></div>`;
      if (n.cases && n.cases.length){
        body += `<div class="bd-cases">` + n.cases.map(c=>`<div class="mono">• ${esc(c)}</div>`).join("") + `</div>`;
      }
      return `<div class="bd-item">${body}</div>`;
    }).join("");
  } else {
    h += `<p class="empty">无归因信息</p>`;
  }
  h += `</div>`;
  const failed = t.failed_tests||[];
  if (failed.length){
    h += `<div class="card"><h3>失败用例明细（${failed.length}）</h3>` + failed.map(f=>
      `<div style="margin-bottom:14px"><div>${badge(f.status)} <span class="mono">${esc(f.name)}</span></div>${f.message?`<pre>${esc(f.message)}</pre>`:""}</div>`).join("") + `</div>`;
  }
  if (t.result && t.result.exception_info){
    const e = t.result.exception_info;
    h += `<div class="card"><h3>异常信息</h3><pre>${esc(JSON.stringify(e, null, 2))}</pre></div>`;
  }
  return h;
}

function renderMain(){
  // 重绘前记录 filter 输入框焦点/光标，避免 oninput 每次按键都被 rebuild 打断
  const ae = document.activeElement;
  const keepFocus = !!(ae && ae.classList && ae.classList.contains("filter"));
  const s0 = keepFocus ? ae.selectionStart : null;
  const s1 = keepFocus ? ae.selectionEnd : null;
  const tabs = [["overview","总览"],["trial","Trial 得分"],["tests","测试用例"],["trace","执行轨迹"],["tools","工具分析"],["patch","Patch"],["badcase","Badcase"]];
  let h = `<div class="tabs">` + tabs.map(([k,l])=>`<button class="tab ${S.tab===k?"on":""}" onclick="selTab('${k}')">${l}</button>`).join("") + `</div>`;
  if (S.tab==="overview") h += aggOverview();
  else if (S.tab==="trial") h += trialOverview();
  else if (S.tab==="tests") h += trialTests();
  else if (S.tab==="trace") h += trialTrace();
  else if (S.tab==="tools") h += trialTools();
  else if (S.tab==="patch") h += trialPatch();
  else h += trialBadcase();
  document.getElementById("main").innerHTML = h;
  if (keepFocus){
    const inp = document.querySelector("#main .filter");
    if (inp){ inp.focus(); try{ inp.setSelectionRange(s0, s1); }catch(e){} }
  }
}

function renderAll(){ renderHeader(); renderNav(); renderMain(); }
renderAll();
</script>
</body>
</html>
'''


def generate_html_report(run_dir, logger=None):
    """为 run 目录生成 index.html，返回生成的文件路径（失败返回 None）。

    内部已捕获异常，失败只告警、不抛出，不阻断评测主流程。
    """
    run_dir = Path(run_dir).resolve()
    try:
        data = {"runs": [_collect_run(run_dir)]}
        payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
        html = TEMPLATE.replace("__DATA__", payload)
        out = run_dir / "index.html"
        out.write_text(html, encoding="utf-8")
        msg = f"HTML report written to {out} ({out.stat().st_size / 1024:.0f} KB)"
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
        return out
    except Exception as e:  # noqa: BLE001
        msg = f"Failed to generate HTML report for {run_dir}: {e}"
        if logger is not None:
            logger.warning(msg)
        else:
            print(msg)
        return None
