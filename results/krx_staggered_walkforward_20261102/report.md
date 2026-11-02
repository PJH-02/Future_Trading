# KRX staggered-universe walk-forward

Counterfactual portfolio period: 2010-02-22 through 2026-09-17 (4080 observed KRX sessions).

Each source's earliest usable date is treated as a hypothetical strategy-development date. These are not actual historical fund launch records. Original strategy rules and S0 sizing are used throughout; no later policy winner is backfilled.

Strategies join the target roster at their assumed development date and first receive capital on a later observed KRX session. EW admits a developed strategy immediately. LP and Convex admit each entrant only after 252 individual and joint known observations; mature teams continue to be optimized while newcomers incubate at zero capital. Before the first trained team is available their capital stays in treasury. The book remains continuous across roster changes.

Runtime 131.4s; candidate cache hits 9683; optimized dates 15300; initial cash warm-up dates 1004; solver failures 16.

## Full-period main results (2.5bp budget-delta proxy)

| Method | Cap | Band | CAGR | MDD | Sharpe | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| CONVEX | 0.05 | 0.05 | 0.0787 | -0.0839 | 1.107 | 0.938 |
| CONVEX | 0.05 | 0.10 | 0.0798 | -0.0839 | 1.112 | 0.951 |
| CONVEX | 0.10 | 0.05 | 0.0909 | -0.1143 | 1.005 | 0.795 |
| CONVEX | 0.10 | 0.10 | 0.0928 | -0.1130 | 1.012 | 0.821 |
| EW | 0.00 | 0.05 | 0.0794 | -0.0454 | 1.573 | 1.752 |
| EW | 0.00 | 0.10 | 0.0822 | -0.0456 | 1.573 | 1.803 |
| LP | 0.05 | 0.05 | 0.0766 | -0.0857 | 1.093 | 0.894 |
| LP | 0.05 | 0.10 | 0.0767 | -0.0857 | 1.094 | 0.895 |
| LP | 0.10 | 0.05 | 0.0891 | -0.1311 | 0.997 | 0.680 |
| LP | 0.10 | 0.10 | 0.0904 | -0.1311 | 1.008 | 0.690 |

## Source and execution limits

- Same-day strategy returns are reused as saved net S0 histories. The allocation model is an account-level daily funding surrogate, not an ETF resize fill or T+2 settlement ledger.
- Rebalance proxy charges 1.5bp fee plus 1bp adverse slippage on absolute per-sleeve capital-budget changes. Treasury/internal cash transfer itself remains free. A no-extra-order-cost sensitivity is included.
- Known source NAs stay NA in fitting histories; execution treats eligible same-day signal gaps as suspended cash. Missing held daynight marks fail.
- Beomgu pre-listing synthetic 2x ETF history is excluded before physical 2x feasibility.

See `strategy_availability.csv`, `universe_timeline.csv`, `solver_status.csv`, yearly/era metric files, and daily audit files for dates and path details.
