# TunnelVision

Continuous-tracking edge ALPR for car wash tunnels. Two RTSP cameras feed a
per-camera pipeline that detects vehicles, tracks them across frames, scores
best frames, queues async Rekor CarCheck calls, and correlates ingress to
egress visits.

See `experiments/tunnelvision/{PRD,ERD,DESIGN,PLAN}.md` for full context.

## Requirements

- DeGirum SDK with local Hailo device (`@local`)
- Hailo-8/8L device with the models recorded in `experiments/tunnelvision/MODELS.md`
- Two RTSP cameras (or test streams)
- `REKOR_SECRET_KEY` in environment (Rekor calls are skipped if missing; visit
  tracking still works)

## Camera role mapping

In the app, `camera_id` is purely logical — `0` = ingress, `1` = egress, always.
The physical "cam1/cam2" labels at each site are independent of the logical
mapping; what matters is which RTSP URL is passed to `--ingress` vs `--egress`.

| Site | `--ingress` (camera_id=0) | `--egress` (camera_id=1) |
|---|---|---|
| `lalaland` | physical cam1 — 192.168.1.121 | physical cam2 — 192.168.1.125 |
| `jajamaica` | physical cam2 — WashiFi Axis 192.168.35.121 | physical cam1 — NVR 192.168.1.50 |

## Quick start — named test scripts

Three wrapper scripts in `experiments/tunnelvision/scripts/` invoke the CLI
with the right URLs, zones, db path, and snapshot dir for each setup:

```bash
# Local cameras (this machine on the lalaland LAN)
bash experiments/tunnelvision/scripts/run_lalaland.sh

# Advance Car Wash, Jamaica Queens — direct (run from the Pi at the site)
bash experiments/tunnelvision/scripts/run_jajamaica.sh

# Advance Car Wash, Jamaica Queens — remote (via SSH+cloudflared tunnel)
bash experiments/tunnelvision/scripts/run_jajamaica_via_tunnel.sh
```

For the **via-tunnel** variant, open the SSH port-forward in a separate
terminal first (leave it running for the duration of the test):

```bash
ssh -o ProxyCommand='cloudflared access ssh --hostname %h' \
    -L 8121:192.168.35.121:554 \
    -L 8150:192.168.1.50:554 \
    -N td-pi@ssh-metal-pi.tuxedodrive.dev
```

`-N` opens forward-only (no shell). The metal-pi has eth1 to the WashiFi
subnet and direct LAN access to the NVR, so it can route both forwards.

Each script writes to a site-specific SQLite db (`tunnelvision-lalaland.db` /
`tunnelvision-jajamaica.db`) and snapshot dir (`tv_snapshots/lalaland/` /
`tv_snapshots/jajamaica/`). The site-specific zones polygons live in
`experiments/tunnelvision/zones-{lalaland,jajamaica}.json`.

**Polygon provenance:**

- `zones-jajamaica.json` — `lpr_zone` and `ingress_gate` polygons are the
  production WashiFi cam2 polygon from `td-edge` deploy on 2026-04-29
  (`docs/plans/2026-04-29-cam2-alpr-latest-frame.md`). `egress_gate` is a
  full-frame placeholder; tune against the NVR ch801 feed at first live test.
- `zones-lalaland.json` — full-frame placeholders. We do not have tuned
  polygons for the local 192.168.1.121 / 192.168.1.125 cameras anywhere in
  the codebase. Tune against actual frames at first live test before relying
  on visit data.

## Quick start — raw CLI

```bash
source setup_env.sh
export REKOR_SECRET_KEY=<your_key>
python3 -m hailo_apps.python.standalone_apps.tunnelvision.tunnelvision \
  --ingress 'rtsp://bowtie:dieformalwear99!@192.168.1.121:554/media/live/1/1' \
  --egress  'rtsp://admin:123456@192.168.1.125:554/media/live/1/1' \
  --zones   experiments/tunnelvision/zones-lalaland.json \
  --db      tunnelvision.db \
  --snapshot-dir tv_snapshots \
  --debug
```

The `!` in the lalaland ingress URL must be inside single quotes — bash history
expansion otherwise mangles it.

## Verification queries

```sql
SELECT status, COUNT(*) FROM visits GROUP BY status;
SELECT COUNT(*), SUM(actual_credit_cost) FROM rekor_carcheck_requests
  WHERE request_status = 'succeeded';
SELECT AVG(tunnel_duration_sec) FROM visits WHERE status = 'completed';
```
