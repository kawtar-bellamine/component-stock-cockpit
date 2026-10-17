-- KPIs computed in SQL (DuckDB syntax, standard apart from date functions) on two tables:
--   demand(part VARCHAR, month VARCHAR 'YYYY-MM', qty DOUBLE)
--   recommendations(part, demand_class, abc_units, demand_last_12m, months_since_last_demand,
--                   forecast_per_month, order_up_to_level, on_hand, on_order, proposed_order, flags)

-- 1. Demand by year
CREATE OR REPLACE VIEW demand_by_year AS
SELECT substr(month, 1, 4) AS year, SUM(qty) AS units, COUNT(DISTINCT part) AS parts_with_rows
FROM demand GROUP BY 1 ORDER BY 1;

-- 2. Parts per demand class, with their share of last-12-month demand
CREATE OR REPLACE VIEW class_summary AS
SELECT demand_class, COUNT(*) AS parts, SUM(demand_last_12m) AS units_last_12m,
       SUM(demand_last_12m) / SUM(SUM(demand_last_12m)) OVER () AS share_of_units
FROM recommendations GROUP BY 1 ORDER BY parts DESC;

-- 3. Parts without demand for 12 months that still hold stock (review before any purchase)
CREATE OR REPLACE VIEW dormant_with_stock AS
SELECT part, on_hand, months_since_last_demand
FROM recommendations
WHERE demand_last_12m = 0 AND on_hand > 0
ORDER BY on_hand DESC;

-- 4. Purchase proposal totals by ABC class
CREATE OR REPLACE VIEW orders_by_abc AS
SELECT abc_units, COUNT(*) FILTER (WHERE proposed_order > 0) AS parts_to_order, SUM(proposed_order) AS units
FROM recommendations GROUP BY 1 ORDER BY 1;
