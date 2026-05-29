# Claude Code Instructions

## File Writing

- Always write markdown files using Bash (`cat > file.md << 'EOF' ... EOF`) rather than the Write tool.

## README Maintenance

- After any code change, check whether it invalidates anything in README.md (API signatures, default values, CLI flags, cache behaviour, data sources). If so, update README.md in the same session using Bash.

## Future Work (Tracked)

- **GSI congestion signal — Phase 2:** Add LMP component-based extraction (energy/congestion/loss decomposition from gridstatus) AND promote inter-zonal spread to a separate 5th GSI sub-signal with rebalanced weights. Deferred until the spread signal is validated in backtesting. See `docs/superpowers/specs/2026-05-28-congestion-spread-design.md`.
