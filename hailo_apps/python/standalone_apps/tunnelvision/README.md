# TunnelVision

Continuous-tracking edge ALPR for car wash tunnels. Two RTSP cameras
(cam0 ingress, cam1 egress) feed a per-camera pipeline that detects vehicles,
tracks them across frames, scores best frames, queues async Rekor CarCheck
calls, and correlates ingress to egress visits.

See `experiments/tunnelvision/{PRD,ERD,DESIGN,PLAN}.md` for full context.

## Requirements

- DeGirum SDK with local Hailo device (`@local`)
- Hailo-8/8L device with the models recorded in `experiments/tunnelvision/MODELS.md`
- Two RTSP cameras (or test streams)
- `REKOR_SECRET_KEY` in environment

## Quick start

```bash
source setup_env.sh
export REKOR_SECRET_KEY=<your_key>
python3 -m hailo_apps.python.standalone_apps.tunnelvision.tunnelvision \
  --ingress 'rtsp://bowtie:dieformalwear99!@192.168.1.121:554/media/live/1/1' \
  --egress  'rtsp://admin:123456@192.168.1.125:554/media/live/1/1' \
  --zones   experiments/tunnelvision/zones.json \
  --db      tunnelvision.db \
  --snapshot-dir tv_snapshots \
  --debug
```

The `!` in the ingress URL must be inside single quotes — bash history expansion
otherwise mangles it.

## Verification queries

```sql
SELECT status, COUNT(*) FROM visits GROUP BY status;
SELECT COUNT(*), SUM(actual_credit_cost) FROM rekor_carcheck_requests
  WHERE request_status = 'succeeded';
SELECT AVG(tunnel_duration_sec) FROM visits WHERE status = 'completed';
```
