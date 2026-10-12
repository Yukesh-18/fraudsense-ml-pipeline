DROP VIEW IF EXISTS features;

CREATE VIEW features AS
SELECT
    txn_id,
    card_id,
    ts_unix,
    amount,
    is_foreign,
    hour,
    is_fraud,

-- how many earlier transactions on this card in the last 1 hour / 24 hours
COUNT(*) OVER (
    PARTITION BY
        card_id
    ORDER BY ts_unix RANGE BETWEEN 3600 PRECEDING
        AND 1 PRECEDING
) AS txn_count_1h,
COUNT(*) OVER (
    PARTITION BY
        card_id
    ORDER BY ts_unix RANGE BETWEEN 86400 PRECEDING
        AND 1 PRECEDING
) AS txn_count_24h,

-- this card's usual spend over the previous 30 days
COALESCE(
    AVG(amount) OVER (
        PARTITION BY
            card_id
        ORDER BY ts_unix RANGE BETWEEN 2592000 PRECEDING
            AND 1 PRECEDING
    ),
    amount
) AS avg_amount_30d
FROM transactions;