# Power BI kit

Files written by `python experiments/run_all.py`:

| File | Content |
|---|---|
| `fact_demand.csv` | real monthly demand, one row per part and month |
| `dim_part.csv` | demand class (ADI, CV²), ABC on units |
| `fact_recommendation.csv` | order-up-to level and proposed order per part (stock is **simulated**, see README) |
| `stock_policy_backtest.csv` | fill rate and stock of each policy, lead time and target |

Steps: **Get data > Text/CSV** for each file, type `part` as Text (part numbers must not lose leading zeros),
`month` as Text or Date (first day of month), quantities as Decimal number; relationships from `fact_demand` and
`fact_recommendation` to `dim_part` on `part`; then the measures of `measures.dax`. Check *Units* by year against
`results/demand_by_year.csv` before building pages.

Suggested pages: demand profile by class; stock policy frontier (`stock_in_months_of_demand` on X, `fill_rate`
on Y, one line per method); purchase proposal by ABC class with the flagged parts listed apart.

The DAX has not been run in CI (Power BI Desktop does not run on Linux); the check above is there for that.
