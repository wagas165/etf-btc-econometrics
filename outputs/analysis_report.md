# Analysis report

## Data validation
```json
{
  "rows_total": 473,
  "date_min": "2024-01-11",
  "date_max": "2025-12-02",
  "duplicate_dates": false,
  "rows_finite_flowshock_turnover": 472,
  "premium_unique_nonmissing": 473,
  "missing_by_col": {
    "date": 0,
    "flow_agg_usd_mn": 0,
    "aum_total": 0,
    "dollar_volume_total": 0,
    "flow_agg_usd": 0,
    "aum_total_lag": 1,
    "FlowShock": 1,
    "Turnover": 1,
    "Premium_agg": 0,
    "btc_open_intraday": 2,
    "btc_close_intraday": 2,
    "btc_close_et": 2,
    "btc_prev_close_et": 3,
    "ret_close_close": 3,
    "ret_intraday": 2,
    "ret_overnight": 3,
    "rv_intraday": 2,
    "rv_overnight": 2,
    "is_fomc": 0,
    "is_cpi": 0,
    "FlowShock_frac": 1
  }
}
```

## Baseline: static OLS (compact)

| window          | spec          |         beta |          se |          t |        p |   n |          r2 |
|:----------------|:--------------|-------------:|------------:|-----------:|---------:|----:|------------:|
| ret_close_close | simple        | -3.49044e-09 | 1.82333e-08 | -0.191432  | 0.848187 | 469 | 4.50025e-05 |
| ret_close_close | with_controls | -9.24078e-09 | 1.85819e-08 | -0.497301  | 0.618977 | 469 | 0.0252469   |
| ret_overnight   | simple        | -5.00749e-09 | 1.0306e-08  | -0.485881  | 0.627051 | 469 | 0.000171986 |
| ret_overnight   | with_controls | -9.69099e-09 | 9.16954e-09 | -1.05687   | 0.290572 | 469 | 0.0579223   |
| ret_intraday    | simple        |  1.51706e-09 | 1.08603e-08 |  0.139688  | 0.888906 | 469 | 1.66918e-05 |
| ret_intraday    | with_controls |  4.50217e-10 | 1.17156e-08 |  0.0384288 | 0.969346 | 469 | 0.0195175   |

## Baseline: event study CAR (thresholds)

| label           |   threshold |   h |   n_events_used |    avg_CAR |     se_CAR |        t |   thr_value |   n_events_total |
|:----------------|------------:|----:|----------------:|-----------:|-----------:|---------:|------------:|-----------------:|
| baseline        |        0.95 |   1 |              24 | 0.00468567 | 0.00665245 | 0.704352 |     42797.8 |               24 |
| baseline        |        0.95 |   2 |              24 | 0.0114665  | 0.00768068 | 1.49291  |     42797.8 |               24 |
| baseline        |        0.95 |   5 |              23 | 0.0265899  | 0.0124131  | 2.14208  |     42797.8 |               24 |
| placebo_shift10 |        0.95 |   1 |              23 | 0.0138899  | 0.00673658 | 2.06187  |     42797.8 |               24 |
| placebo_shift10 |        0.95 |   2 |              23 | 0.0272638  | 0.00971695 | 2.8058   |     42797.8 |               24 |
| placebo_shift10 |        0.95 |   5 |              23 | 0.0611135  | 0.0160008  | 3.8194   |     42797.8 |               24 |
| baseline        |        0.9  |   1 |              47 | 0.00837796 | 0.00516743 | 1.6213   |     16093.2 |               47 |
| baseline        |        0.9  |   2 |              47 | 0.0165067  | 0.00651421 | 2.53395  |     16093.2 |               47 |
| baseline        |        0.9  |   5 |              46 | 0.0404083  | 0.0115415  | 3.50114  |     16093.2 |               47 |
| placebo_shift10 |        0.9  |   1 |              46 | 0.00856366 | 0.00492144 | 1.74007  |     16093.2 |               47 |
| placebo_shift10 |        0.9  |   2 |              46 | 0.0154734  | 0.00645255 | 2.39803  |     16093.2 |               47 |
| placebo_shift10 |        0.9  |   5 |              46 | 0.0342504  | 0.011652   | 2.93945  |     16093.2 |               47 |

## IV: instruments used

```json
{
  "candidates": [
    "Turnover",
    "Premium_agg"
  ],
  "used": [
    "Turnover",
    "Premium_agg"
  ]
}
```

## Robustness

See robustness_report.md and the robustness CSV outputs for details.
