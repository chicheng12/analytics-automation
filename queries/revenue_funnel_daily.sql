-- Revenue Funnel Daily Metrics
-- ==============================
-- Parameterized SQL template for the revenue funnel monitor.
-- Computes daily counts for: intentful visitors, requests, projects, contacts, revenue
-- with optional channel breakdown using the 4-channel taxonomy:
--   O&O = Direct App, Direct Web, Paid Other O&O, EM
--   SEM = SEM, Paid Other
--   SEO = SEO
--   Partnership = Partnership
--
-- Parameters (injected by monitor_engine.py):
--   {date_filter}            -- e.g., "BETWEEN '2026-01-01' AND '2026-02-05'"
--   {dimension_col_iv}       -- IV table dimension expression
--   {dimension_col}          -- Requests table dimension expression
--   {dimension_col_project}  -- Projects table dimension expression
--   {dimension_col_contact}  -- Contacts table dimension expression
--   {dimension_col_revenue}  -- Revenue table dimension expression
--
-- All timestamps use PT timezone as per business convention.
-- Revenue uses UNION ALL of both sources (attributed + ttod).
-- Projects identified by request_pk where project_created_time IS NOT NULL.

WITH
channel_map AS (
    SELECT 'Direct App' AS segment_detailed, 'O&O' AS channel UNION ALL
    SELECT 'Direct Web', 'O&O' UNION ALL
    SELECT 'Paid Other O&O', 'O&O' UNION ALL
    SELECT 'EM', 'O&O' UNION ALL
    SELECT 'SEM', 'SEM' UNION ALL
    SELECT 'Paid Other', 'SEM' UNION ALL
    SELECT 'SEO', 'SEO' UNION ALL
    SELECT 'Partnership', 'Partnership'
),

-- Intentful Visitors
iv_daily AS (
    SELECT
        iv.visit_date AS metric_date,
        {dimension_col_iv} AS dimension,
        COUNT(*) AS intentful_visitors
    FROM `tt-dp-prod.sot_analytics.intentful_visits` AS iv
    LEFT JOIN channel_map cm_iv ON iv.tackboard_segment_detailed = cm_iv.segment_detailed
    WHERE iv.visit_date {date_filter}
    GROUP BY 1, 2
),

-- Requests
requests_daily AS (
    SELECT
        DATE(r.request_created_time, 'America/Los_Angeles') AS metric_date,
        {dimension_col} AS dimension,
        COUNT(DISTINCT r.request_pk) AS requests
    FROM `tt-dp-prod.sot_analytics.requests` AS r
    LEFT JOIN channel_map cm ON r.tackboard_segment_detailed = cm.segment_detailed
    WHERE r.request_created_time IS NOT NULL
        AND DATE(r.request_created_time, 'America/Los_Angeles') {date_filter}
    GROUP BY 1, 2
),

-- Projects (identified by request_pk where project was created)
projects_daily AS (
    SELECT
        DATE(p.project_created_time, 'America/Los_Angeles') AS metric_date,
        {dimension_col_project} AS dimension,
        COUNT(DISTINCT p.request_pk) AS projects
    FROM `tt-dp-prod.sot_analytics.projects` AS p
    LEFT JOIN `tt-dp-prod.sot_analytics.requests` AS r
        ON p.request_pk = r.request_pk
    LEFT JOIN channel_map cm ON r.tackboard_segment_detailed = cm.segment_detailed
    WHERE p.project_created_time IS NOT NULL
        AND DATE(p.project_created_time, 'America/Los_Angeles') {date_filter}
    GROUP BY 1, 2
),

-- Contacts
contacts_daily AS (
    SELECT
        DATE(c.contact_created_time, 'America/Los_Angeles') AS metric_date,
        {dimension_col_contact} AS dimension,
        COUNT(DISTINCT c.contact_pk) AS contacts
    FROM `tt-dp-prod.sot_analytics.contacts` AS c
    LEFT JOIN `tt-dp-prod.sot_analytics.requests` AS r
        ON c.request_pk = r.request_pk
    LEFT JOIN channel_map cm ON r.tackboard_segment_detailed = cm.segment_detailed
    WHERE c.contact_created_time IS NOT NULL
        AND DATE(c.contact_created_time, 'America/Los_Angeles') {date_filter}
    GROUP BY 1, 2
),

-- Revenue (UNION ALL of both sources, then aggregate)
revenue_combined AS (
    SELECT
        DATE(rev.transaction_timestamp, 'America/Los_Angeles') AS metric_date,
        {dimension_col_revenue} AS dimension,
        rev.gross_revenue AS revenue
    FROM `tt-dp-prod.sot_intermediate.attributed_requests_pro_revenue` AS rev
    LEFT JOIN `tt-dp-prod.sot_analytics.requests` AS r
        ON rev.request_pk = r.request_pk
    LEFT JOIN channel_map cm ON r.tackboard_segment_detailed = cm.segment_detailed
    WHERE DATE(rev.transaction_timestamp, 'America/Los_Angeles') {date_filter}

    UNION ALL

    SELECT
        DATE(rev.ts, 'America/Los_Angeles') AS metric_date,
        {dimension_col_revenue} AS dimension,
        rev.net_revenue
            + IFNULL(rev.refund, 0)
            + IFNULL(rev.payout, 0)
            + IFNULL(rev.transfer_reversal, 0)
            + IFNULL(rev.cancellation_revenue, 0)
            + IFNULL(rev.transfer_reversal_failed, 0)
        AS revenue
    FROM `tt-dp-prod.sot_intermediate.ttod_revenue` AS rev
    LEFT JOIN `tt-dp-prod.sot_analytics.requests` AS r
        ON rev.request_pk = r.request_pk
    LEFT JOIN channel_map cm ON r.tackboard_segment_detailed = cm.segment_detailed
    WHERE DATE(rev.ts, 'America/Los_Angeles') {date_filter}
),
revenue_daily AS (
    SELECT metric_date, dimension, SUM(revenue) AS revenue
    FROM revenue_combined
    GROUP BY 1, 2
)

-- Combine all metrics via FULL OUTER JOINs
SELECT
    COALESCE(iv.metric_date, req.metric_date, prj.metric_date, con.metric_date, rev.metric_date) AS metric_date,
    COALESCE(iv.dimension, req.dimension, prj.dimension, con.dimension, rev.dimension) AS dimension,
    COALESCE(iv.intentful_visitors, 0) AS intentful_visitors,
    COALESCE(req.requests, 0) AS requests,
    COALESCE(prj.projects, 0) AS projects,
    COALESCE(con.contacts, 0) AS contacts,
    COALESCE(rev.revenue, 0) AS revenue
FROM iv_daily iv
FULL OUTER JOIN requests_daily req
    ON iv.metric_date = req.metric_date AND iv.dimension = req.dimension
FULL OUTER JOIN projects_daily prj
    ON COALESCE(iv.metric_date, req.metric_date) = prj.metric_date
    AND COALESCE(iv.dimension, req.dimension) = prj.dimension
FULL OUTER JOIN contacts_daily con
    ON COALESCE(iv.metric_date, req.metric_date, prj.metric_date) = con.metric_date
    AND COALESCE(iv.dimension, req.dimension, prj.dimension) = con.dimension
FULL OUTER JOIN revenue_daily rev
    ON COALESCE(iv.metric_date, req.metric_date, prj.metric_date, con.metric_date) = rev.metric_date
    AND COALESCE(iv.dimension, req.dimension, prj.dimension, con.dimension) = rev.dimension
WHERE COALESCE(iv.metric_date, req.metric_date, prj.metric_date, con.metric_date, rev.metric_date) IS NOT NULL
ORDER BY metric_date, dimension;
