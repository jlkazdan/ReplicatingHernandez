"""Simple terminal dashboard + CSV + HTML for W&B runs in hernandez-replication project."""

import csv
import os
import wandb
from datetime import datetime, timezone

ENTITY = "jchud-stanford-university"
PROJECT = "hernandez-replication"
WANDB_BASE = f"https://wandb.ai/{ENTITY}/{PROJECT}/runs"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "dashboard")
os.makedirs(OUT_DIR, exist_ok=True)

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

all_runs = []
now = datetime.now(timezone.utc)

for run in runs:
    c = run.config
    mc = c.get("model_config", {})
    dc = c.get("data_config", {})

    model_name = mc.get("model_name", "")
    model_size = model_name.split("/")[-1].replace("Qwen3-", "") if "/" in model_name else "?"

    num_repeats = dc.get("num_repeats", None)
    rep_budget = dc.get("repetition_budget", None)
    if num_repeats is not None and num_repeats > 1:
        repeated = "repeated"
        rep_detail = f"x{num_repeats}, {int(rep_budget*100)}% budget" if rep_budget else f"x{num_repeats}"
    else:
        repeated = "non-repeated"
        rep_detail = ""
        num_repeats = num_repeats or 1

    start = datetime.fromisoformat(run.created_at.replace("Z", "+00:00"))

    if run.state == "finished":
        runtime_sec = run.summary.get("_wandb", {}).get("runtime", None) if run.summary else None
        if runtime_sec is not None:
            h, rem = divmod(int(runtime_sec), 3600)
            m, _ = divmod(rem, 60)
            duration = f"{h}h {m}m"
            from datetime import timedelta
            end_time = start + timedelta(seconds=runtime_sec)
            ago = now - end_time
            ago_h, ago_rem = divmod(int(ago.total_seconds()), 3600)
            ago_m, _ = divmod(ago_rem, 60)
            if ago_h >= 24:
                finished_ago = f"{ago_h // 24}d {ago_h % 24}h ago"
            else:
                finished_ago = f"{ago_h}h {ago_m}m ago"
        else:
            duration = "N/A"
            finished_ago = "N/A"
    elif run.state == "running":
        elapsed = now - start
        h, rem = divmod(int(elapsed.total_seconds()), 3600)
        m, _ = divmod(rem, 60)
        duration = f"{h}h {m}m (running)"
        finished_ago = ""
    elif run.state in ("failed", "crashed"):
        elapsed = now - start
        h, rem = divmod(int(elapsed.total_seconds()), 3600)
        m, _ = divmod(rem, 60)
        duration = f"{h}h {m}m ago"
        finished_ago = ""
        # only keep if created in last 120 hours
        if elapsed.total_seconds() > 120 * 3600:
            continue
    else:
        continue

    all_runs.append({
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "model_size": model_size,
        "repeated": repeated,
        "num_repeats": num_repeats,
        "rep_budget": f"{int(rep_budget*100)}%" if rep_budget else "",
        "duration": duration,
        "finished_ago": finished_ago,
        "created": run.created_at,
        "url": f"{WANDB_BASE}/{run.id}",
    })

# --- Terminal output ---
running = [r for r in all_runs if r["state"] == "running"]
finished = [r for r in all_runs if r["state"] == "finished"]
crashed = [r for r in all_runs if r["state"] in ("failed", "crashed")]

print("=" * 100)
print(f"  W&B Dashboard: {ENTITY}/{PROJECT}")
print(f"  Time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 100)

for label, subset in [("RUNNING", running), ("FINISHED", finished), ("CRASHED/FAILED (last 5 days)", crashed)]:
    print(f"\n  {label} ({len(subset)} runs)")
    print("-" * 140)
    if subset:
        print(f"  {'Name':<30} {'Model':<10} {'Type':<15} {'Repeats':<10} {'Duration':<15} {'Finished':<15} {'W&B Link'}")
        print("  " + "-" * 145)
        for r in subset:
            print(f"  {r['name']:<30} {r['model_size']:<10} {r['repeated']:<15} {str(r['num_repeats']):<10} {r['duration']:<15} {r['finished_ago']:<15} {r['url']}")
    else:
        print("  None.")

print(f"\n{'=' * 100}")
print(f"  Total: {len(running)} running, {len(finished)} finished, {len(crashed)} crashed/failed (last 5d)")
print("=" * 100)

# --- CSV ---
csv_path = os.path.join(OUT_DIR, "runs.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Name", "State", "Model", "Repeated", "Num Repeats", "Rep Budget", "Duration", "Finished", "Created", "W&B Link"])
    for r in all_runs:
        w.writerow([r["name"], r["state"], r["model_size"], r["repeated"],
                     r["num_repeats"], r["rep_budget"], r["duration"], r["finished_ago"], r["created"], r["url"]])
print(f"\nCSV saved: {csv_path}")

# --- HTML ---
html_path = os.path.join(OUT_DIR, "runs.html")

def state_badge(state):
    color = "#22c55e" if state == "running" else "#6b7280"
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">{state}</span>'

rows_html = ""
for r in all_runs:
    rows_html += f"""<tr>
  <td><a href="{r['url']}" target="_blank">{r['name']}</a></td>
  <td>{state_badge(r['state'])}</td>
  <td>{r['model_size']}</td>
  <td>{r['repeated']}</td>
  <td>{r['num_repeats']}</td>
  <td>{r['rep_budget']}</td>
  <td>{r['duration']}</td>
  <td>{r['created']}</td>
</tr>"""

html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>W&B Runs — {PROJECT}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 20px; background: #f9fafb; }}
  h1 {{ font-size: 20px; }}
  .meta {{ color: #6b7280; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; font-size: 14px; }}
  th {{ background: #f3f4f6; font-weight: 600; position: sticky; top: 0; cursor: pointer; }}
  th:hover {{ background: #e5e7eb; }}
  tr:hover {{ background: #f9fafb; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .stats {{ display: flex; gap: 20px; margin-bottom: 16px; }}
  .stat {{ background: #fff; padding: 12px 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stat-num {{ font-size: 24px; font-weight: 700; }}
  .stat-label {{ color: #6b7280; font-size: 13px; }}
</style>
</head><body>
<h1>W&B Runs — {ENTITY}/{PROJECT}</h1>
<div class="meta">Updated: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
<div class="stats">
  <div class="stat"><div class="stat-num">{len(running)}</div><div class="stat-label">Running</div></div>
  <div class="stat"><div class="stat-num">{len(finished)}</div><div class="stat-label">Finished</div></div>
  <div class="stat"><div class="stat-num">{len(all_runs)}</div><div class="stat-label">Total</div></div>
</div>
<table id="runs">
<thead><tr>
  <th onclick="sortTable(0)">Name</th>
  <th onclick="sortTable(1)">State</th>
  <th onclick="sortTable(2)">Model</th>
  <th onclick="sortTable(3)">Repeated</th>
  <th onclick="sortTable(4)">Num Repeats</th>
  <th onclick="sortTable(5)">Rep Budget</th>
  <th onclick="sortTable(6)">Duration</th>
  <th onclick="sortTable(7)">Created</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
<script>
let sortDir = {{}};
function sortTable(col) {{
  const table = document.getElementById("runs");
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  sortDir[col] = !sortDir[col];
  rows.sort((a, b) => {{
    let av = a.cells[col].textContent.trim();
    let bv = b.cells[col].textContent.trim();
    let an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return sortDir[col] ? an - bn : bn - an;
    return sortDir[col] ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
</script>
</body></html>"""

with open(html_path, "w") as f:
    f.write(html)
print(f"HTML saved: {html_path}")
print(f"\nOpen in browser: file://{html_path}")
