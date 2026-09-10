# Cross-round Dependency Audit

Status: `CLEANROOM-GATE-PASS`

The clean-room runner calls pure in-memory functions or redirected wrappers. Historical result directories are blocked until the explicit old-vs-new phase.

```json
{
  "old_results_used_as_inputs": false,
  "path_guard_probes": {
    "cleanroom_results_allowed": true,
    "old_task1_snapshot_blocked": true,
    "result_registry_blocked": true,
    "round1_results_blocked": true,
    "round2_results_blocked": true
  },
  "static_disallowed_invocations": [],
  "status": "CLEANROOM-GATE-PASS"
}
```
