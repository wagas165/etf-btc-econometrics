# Step 6 Robustness Summary


## Variants

base, exclude_macro, bull, bear, pre_t1, post_t1, year_2024, year_2025


## Static OLS robustness (with controls, HAC=5)


      variant          window          shock      beta       se         t        p   n       r2
         base ret_close_close FlowShock_frac  0.313645 0.407243  0.770166 0.441202 196 0.006920
         base   ret_overnight FlowShock_frac  0.325021 0.330743  0.982701 0.325755 196 0.017063
         base    ret_intraday FlowShock_frac -0.011376 0.195739 -0.058119 0.953654 196 0.002510
exclude_macro ret_close_close FlowShock_frac  0.311556 0.412834  0.754678 0.450442 192 0.007085
exclude_macro   ret_overnight FlowShock_frac  0.329177 0.328344  1.002536 0.316085 192 0.011228
exclude_macro    ret_intraday FlowShock_frac -0.017621 0.197932 -0.089024 0.929063 192 0.001687
         bull ret_close_close FlowShock_frac -0.427532 0.428364 -0.998058 0.318251 131 0.024722
         bull   ret_overnight FlowShock_frac  0.034194 0.347553  0.098384 0.921627 131 0.014898
         bull    ret_intraday FlowShock_frac -0.461726 0.224286 -2.058651 0.039528 131 0.029129
         bear ret_close_close FlowShock_frac  0.121551 1.586823  0.076600 0.938941  64 0.044582
         bear   ret_overnight FlowShock_frac  0.340454 1.326072  0.256738 0.797381  64 0.035557
         bear    ret_intraday FlowShock_frac -0.218902 0.643160 -0.340355 0.733589  64 0.074001


(Full table: step6_static_ols_robustness.csv)



## LP-IV h=0 robustness


      variant          window  h       beta          se         t        p   n error
         base ret_close_close  0   1.760678    2.888240  0.609602 0.543468 108   NaN
         base   ret_overnight  0   1.356479    1.145516  1.184165 0.239073 108   NaN
         base    ret_intraday  0   0.404199    2.457365  0.164485 0.869672 108   NaN
exclude_macro ret_close_close  0   1.975113    2.955185  0.668355 0.505388 109   NaN
exclude_macro   ret_overnight  0   1.158070    1.214223  0.953754 0.342420 109   NaN
exclude_macro    ret_intraday  0   0.817043    2.349280  0.347784 0.728705 109   NaN
         bull ret_close_close  0   1.250541    4.816416  0.259641 0.795902  75   NaN
         bull   ret_overnight  0   1.028570    1.758112  0.585042 0.560401  75   NaN
         bull    ret_intraday  0   0.221972    3.527905  0.062919 0.950010  75   NaN
         bear ret_close_close  0 -19.568440 2080.793730 -0.009404 0.992568  31   NaN
         bear   ret_overnight  0 -23.477067 2264.357850 -0.010368 0.991807  31   NaN
         bear    ret_intraday  0   3.908627  250.961302  0.015575 0.987693  31   NaN


(Full table: step6_lpiv_h0_robustness.csv)



## HAC lags sensitivity


                model  hac_lags     beta       se        t        p   n instrument     coef         F       r2
static_ols_closeclose         3 0.313645 0.386564 0.811367 0.417155 196        NaN      NaN       NaN      NaN
static_ols_closeclose         5 0.313645 0.407243 0.770166 0.441202 196        NaN      NaN       NaN      NaN
static_ols_closeclose        10 0.313645 0.424668 0.738566 0.460171 196        NaN      NaN       NaN      NaN
          first_stage         3      NaN 0.024108 4.227133      NaN 109   Turnover 0.101909 17.868655 0.379388
          first_stage         5      NaN 0.025608 3.979536      NaN 109   Turnover 0.101909 15.836710 0.379388
          first_stage        10      NaN 0.026744 3.810474      NaN 109   Turnover 0.101909 14.519714 0.379388


## Event study thresholds


          label  threshold  h  n_events_used   avg_CAR   se_CAR         t variant  thr_value  n_events_total
       baseline       0.90  1             20 -0.001336 0.005949 -0.224508    base   0.012754              20
       baseline       0.90  2             20  0.005066 0.008846  0.572624    base   0.012754              20
       baseline       0.90  5             20  0.024907 0.016044  1.552431    base   0.012754              20
placebo_shift10       0.90  1             20  0.000944 0.006816  0.138513    base   0.012754              20
placebo_shift10       0.90  2             20  0.001070 0.008475  0.126208    base   0.012754              20
placebo_shift10       0.90  5             20 -0.001530 0.012305 -0.124356    base   0.012754              20
       baseline       0.95  1             10 -0.002119 0.008927 -0.237311    base   0.016308              10
       baseline       0.95  2             10  0.000316 0.011848  0.026701    base   0.016308              10
       baseline       0.95  5             10  0.029194 0.022869  1.276570    base   0.016308              10
placebo_shift10       0.95  1             10 -0.003445 0.008491 -0.405710    base   0.016308              10
placebo_shift10       0.95  2             10  0.003548 0.011810  0.300442    base   0.016308              10
placebo_shift10       0.95  5             10 -0.009367 0.015505 -0.604147    base   0.016308              10