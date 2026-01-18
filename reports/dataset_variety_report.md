## Dataset variety report

- **Dataset path**: `data/processed/train`
- **Generated at**: `2026-01-18T15:55:23`

## Overall summary

- **n_envs**: `20`
- **mean_pairwise_coverage**: `0.4277836259920635`
- **mean_episode_embedding_knn**: `1.5712341947239366`
- **mean_shared_state_frac**: `0.6585317460317459`
- **mean_intersection_over_union**: `0.4477169770729608`
- **mean_intra_over_inter**: `0.1534739388583564`

## Per-environment report

### McCandless_2008_Chapter_13

- **n_episodes**: `40`
- **env_names**: `40` items

```
McCandless_2008_Chapter_13_config_0.parquet, McCandless_2008_Chapter_13_config_1.parquet, McCandless_2008_Chapter_13_config_10.parquet, McCandless_2008_Chapter_13_config_11.parquet, McCandless_2008_Chapter_13_config_12.parquet, McCandless_2008_Chapter_13_config_13.parquet, McCandless_2008_Chapter_13_config_14.parquet, McCandless_2008_Chapter_13_config_15.parquet, McCandless_2008_Chapter_13_config_16.parquet, McCandless_2008_Chapter_13_config_17.parquet, McCandless_2008_Chapter_13_config_18.parquet, McCandless_2008_Chapter_13_config_19.parquet, McCandless_2008_Chapter_13_config_2.parquet, McCandless_2008_Chapter_13_config_20.parquet, McCandless_2008_Chapter_13_config_21.parquet, McCandless_2008_Chapter_13_config_22.parquet, McCandless_2008_Chapter_13_config_23.parquet, McCandless_2008_Chapter_13_config_24.parquet, McCandless_2008_Chapter_13_config_25.parquet, McCandless_2008_Chapter_13_config_26.parquet, McCandless_2008_Chapter_13_config_27.parquet, McCandless_2008_Chapter_13_config_28.parquet, McCandless_2008_Chapter_13_config_29.parquet, McCandless_2008_Chapter_13_config_3.parquet, McCandless_2008_Chapter_13_config_30.parquet, McCandless_2008_Chapter_13_config_31.parquet, McCandless_2008_Chapter_13_config_32.parquet, McCandless_2008_Chapter_13_config_33.parquet, McCandless_2008_Chapter_13_config_34.parquet, McCandless_2008_Chapter_13_config_35.parquet, McCandless_2008_Chapter_13_config_36.parquet, McCandless_2008_Chapter_13_config_37.parquet, McCandless_2008_Chapter_13_config_38.parquet, McCandless_2008_Chapter_13_config_39.parquet, McCandless_2008_Chapter_13_config_4.parquet, McCandless_2008_Chapter_13_config_5.parquet, McCandless_2008_Chapter_13_config_6.parquet, McCandless_2008_Chapter_13_config_7.parquet, McCandless_2008_Chapter_13_config_8.parquet, McCandless_2008_Chapter_13_config_9.parquet
```
- **state_names**: `14` items

```
W, r, C, K, H, M, P, {P^*}, g, \lambda, B, {r^f}, e, X
```
- **action_names**: `1` items

```
Growth Rate Of Money Stock
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.11571428571428573`
- **mean_pairwise_coverage**: `0.8842857142857142`
- **mean_episode_embedding_knn**: `0.0`
- **nearest_env**: `McCandless_2008_Chapter_9`
- **shared_state_frac**: `0.6428571428571429`
- **intersection_over_union**: `0.6`
- **intra_over_inter**: `0.0`

### RBC_state_dependent_GIRF

- **n_episodes**: `40`
- **env_names**: `40` items

```
RBC_state_dependent_GIRF_config_0.parquet, RBC_state_dependent_GIRF_config_1.parquet, RBC_state_dependent_GIRF_config_10.parquet, RBC_state_dependent_GIRF_config_11.parquet, RBC_state_dependent_GIRF_config_12.parquet, RBC_state_dependent_GIRF_config_13.parquet, RBC_state_dependent_GIRF_config_14.parquet, RBC_state_dependent_GIRF_config_15.parquet, RBC_state_dependent_GIRF_config_16.parquet, RBC_state_dependent_GIRF_config_17.parquet, RBC_state_dependent_GIRF_config_18.parquet, RBC_state_dependent_GIRF_config_19.parquet, RBC_state_dependent_GIRF_config_2.parquet, RBC_state_dependent_GIRF_config_20.parquet, RBC_state_dependent_GIRF_config_21.parquet, RBC_state_dependent_GIRF_config_22.parquet, RBC_state_dependent_GIRF_config_23.parquet, RBC_state_dependent_GIRF_config_24.parquet, RBC_state_dependent_GIRF_config_25.parquet, RBC_state_dependent_GIRF_config_26.parquet, RBC_state_dependent_GIRF_config_27.parquet, RBC_state_dependent_GIRF_config_28.parquet, RBC_state_dependent_GIRF_config_29.parquet, RBC_state_dependent_GIRF_config_3.parquet, RBC_state_dependent_GIRF_config_30.parquet, RBC_state_dependent_GIRF_config_31.parquet, RBC_state_dependent_GIRF_config_32.parquet, RBC_state_dependent_GIRF_config_33.parquet, RBC_state_dependent_GIRF_config_34.parquet, RBC_state_dependent_GIRF_config_35.parquet, RBC_state_dependent_GIRF_config_36.parquet, RBC_state_dependent_GIRF_config_37.parquet, RBC_state_dependent_GIRF_config_38.parquet, RBC_state_dependent_GIRF_config_39.parquet, RBC_state_dependent_GIRF_config_4.parquet, RBC_state_dependent_GIRF_config_5.parquet, RBC_state_dependent_GIRF_config_6.parquet, RBC_state_dependent_GIRF_config_7.parquet, RBC_state_dependent_GIRF_config_8.parquet, RBC_state_dependent_GIRF_config_9.parquet
```
- **state_names**: `9` items

```
{y}, {c}, {k}, {l}, {z}, {\hat g}, {r}, {w}, {i}
```
- **action_names**: `1` items

```
Government Spending
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.15222222222222223`
- **mean_pairwise_coverage**: `0.8477777777777777`
- **mean_episode_embedding_knn**: `0.0`
- **nearest_env**: `RBC_news_shock_model`
- **shared_state_frac**: `0.8888888888888888`
- **intersection_over_union**: `0.8888888888888888`
- **intra_over_inter**: `4.301594713252976e-09`

### Faia_2008

- **n_episodes**: `40`
- **env_names**: `40` items

```
Faia_2008_config_0.parquet, Faia_2008_config_1.parquet, Faia_2008_config_10.parquet, Faia_2008_config_11.parquet, Faia_2008_config_12.parquet, Faia_2008_config_13.parquet, Faia_2008_config_14.parquet, Faia_2008_config_15.parquet, Faia_2008_config_16.parquet, Faia_2008_config_17.parquet, Faia_2008_config_18.parquet, Faia_2008_config_19.parquet, Faia_2008_config_2.parquet, Faia_2008_config_20.parquet, Faia_2008_config_21.parquet, Faia_2008_config_22.parquet, Faia_2008_config_23.parquet, Faia_2008_config_24.parquet, Faia_2008_config_25.parquet, Faia_2008_config_26.parquet, Faia_2008_config_27.parquet, Faia_2008_config_28.parquet, Faia_2008_config_29.parquet, Faia_2008_config_3.parquet, Faia_2008_config_30.parquet, Faia_2008_config_31.parquet, Faia_2008_config_32.parquet, Faia_2008_config_33.parquet, Faia_2008_config_34.parquet, Faia_2008_config_35.parquet, Faia_2008_config_36.parquet, Faia_2008_config_37.parquet, Faia_2008_config_38.parquet, Faia_2008_config_39.parquet, Faia_2008_config_4.parquet, Faia_2008_config_5.parquet, Faia_2008_config_6.parquet, Faia_2008_config_7.parquet, Faia_2008_config_8.parquet, Faia_2008_config_9.parquet
```
- **state_names**: `24` items

```
{\Lambda}, {c}, {R}, {\pi}, {\theta}, {v}, {u}, {m}, {q}, {n}, {y^{gross}}, {y^{net}}, {\mu}, {z}, {mc}, {w}, {g}, Government Spending Shock, {\log y}, {\log v}, {\log w}, {\log u}, {\log \theta}, {\log \pi}
```
- **action_names**: `1` items

```
Nominal Interest Rate
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.16266666666666668`
- **mean_pairwise_coverage**: `0.8373333333333333`
- **mean_episode_embedding_knn**: `0.0`
- **nearest_env**: `RBC_news_shock_model`
- **shared_state_frac**: `0.125`
- **intersection_over_union**: `0.10344827586206896`
- **intra_over_inter**: `3.0017482427892376e-09`

### Hansen_1985

- **n_episodes**: `40`
- **env_names**: `40` items

```
Hansen_1985_config_0.parquet, Hansen_1985_config_1.parquet, Hansen_1985_config_10.parquet, Hansen_1985_config_11.parquet, Hansen_1985_config_12.parquet, Hansen_1985_config_13.parquet, Hansen_1985_config_14.parquet, Hansen_1985_config_15.parquet, Hansen_1985_config_16.parquet, Hansen_1985_config_17.parquet, Hansen_1985_config_18.parquet, Hansen_1985_config_19.parquet, Hansen_1985_config_2.parquet, Hansen_1985_config_20.parquet, Hansen_1985_config_21.parquet, Hansen_1985_config_22.parquet, Hansen_1985_config_23.parquet, Hansen_1985_config_24.parquet, Hansen_1985_config_25.parquet, Hansen_1985_config_26.parquet, Hansen_1985_config_27.parquet, Hansen_1985_config_28.parquet, Hansen_1985_config_29.parquet, Hansen_1985_config_3.parquet, Hansen_1985_config_30.parquet, Hansen_1985_config_31.parquet, Hansen_1985_config_32.parquet, Hansen_1985_config_33.parquet, Hansen_1985_config_34.parquet, Hansen_1985_config_35.parquet, Hansen_1985_config_36.parquet, Hansen_1985_config_37.parquet, Hansen_1985_config_38.parquet, Hansen_1985_config_39.parquet, Hansen_1985_config_4.parquet, Hansen_1985_config_5.parquet, Hansen_1985_config_6.parquet, Hansen_1985_config_7.parquet, Hansen_1985_config_8.parquet, Hansen_1985_config_9.parquet
```
- **state_names**: `9` items

```
c, w, r, y, h, k, i, \lambda, {\frac{y}{h}}
```
- **action_names**: `1` items

```
Investment
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.26111111111111107`
- **mean_pairwise_coverage**: `0.7388888888888889`
- **mean_episode_embedding_knn**: `0.0`
- **nearest_env**: `Caldara_et_al_2012`
- **shared_state_frac**: `0.4444444444444444`
- **intersection_over_union**: `0.23529411764705882`
- **intra_over_inter**: `0.0`

### Caldara_et_al_2012

- **n_episodes**: `40`
- **env_names**: `40` items

```
Caldara_et_al_2012_config_0.parquet, Caldara_et_al_2012_config_1.parquet, Caldara_et_al_2012_config_10.parquet, Caldara_et_al_2012_config_11.parquet, Caldara_et_al_2012_config_12.parquet, Caldara_et_al_2012_config_13.parquet, Caldara_et_al_2012_config_14.parquet, Caldara_et_al_2012_config_15.parquet, Caldara_et_al_2012_config_16.parquet, Caldara_et_al_2012_config_17.parquet, Caldara_et_al_2012_config_18.parquet, Caldara_et_al_2012_config_19.parquet, Caldara_et_al_2012_config_2.parquet, Caldara_et_al_2012_config_20.parquet, Caldara_et_al_2012_config_21.parquet, Caldara_et_al_2012_config_22.parquet, Caldara_et_al_2012_config_23.parquet, Caldara_et_al_2012_config_24.parquet, Caldara_et_al_2012_config_25.parquet, Caldara_et_al_2012_config_26.parquet, Caldara_et_al_2012_config_27.parquet, Caldara_et_al_2012_config_28.parquet, Caldara_et_al_2012_config_29.parquet, Caldara_et_al_2012_config_3.parquet, Caldara_et_al_2012_config_30.parquet, Caldara_et_al_2012_config_31.parquet, Caldara_et_al_2012_config_32.parquet, Caldara_et_al_2012_config_33.parquet, Caldara_et_al_2012_config_34.parquet, Caldara_et_al_2012_config_35.parquet, Caldara_et_al_2012_config_36.parquet, Caldara_et_al_2012_config_37.parquet, Caldara_et_al_2012_config_38.parquet, Caldara_et_al_2012_config_39.parquet, Caldara_et_al_2012_config_4.parquet, Caldara_et_al_2012_config_5.parquet, Caldara_et_al_2012_config_6.parquet, Caldara_et_al_2012_config_7.parquet, Caldara_et_al_2012_config_8.parquet, Caldara_et_al_2012_config_9.parquet
```
- **state_names**: `12` items

```
V, y, c, k, i, l, z, s, {E_t(SDF_{t+1})}, \sigma, {E_t(R^k_{t+1})}, {R^f}
```
- **action_names**: `1` items

```
Investment
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.2675`
- **mean_pairwise_coverage**: `0.7324999999999999`
- **mean_episode_embedding_knn**: `0.0`
- **nearest_env**: `Hansen_1985`
- **shared_state_frac**: `0.3333333333333333`
- **intersection_over_union**: `0.23529411764705882`
- **intra_over_inter**: `0.0`

### Collard_2001_example1

- **n_episodes**: `40`
- **env_names**: `40` items

```
Collard_2001_example1_config_0.parquet, Collard_2001_example1_config_1.parquet, Collard_2001_example1_config_10.parquet, Collard_2001_example1_config_11.parquet, Collard_2001_example1_config_12.parquet, Collard_2001_example1_config_13.parquet, Collard_2001_example1_config_14.parquet, Collard_2001_example1_config_15.parquet, Collard_2001_example1_config_16.parquet, Collard_2001_example1_config_17.parquet, Collard_2001_example1_config_18.parquet, Collard_2001_example1_config_19.parquet, Collard_2001_example1_config_2.parquet, Collard_2001_example1_config_20.parquet, Collard_2001_example1_config_21.parquet, Collard_2001_example1_config_22.parquet, Collard_2001_example1_config_23.parquet, Collard_2001_example1_config_24.parquet, Collard_2001_example1_config_25.parquet, Collard_2001_example1_config_26.parquet, Collard_2001_example1_config_27.parquet, Collard_2001_example1_config_28.parquet, Collard_2001_example1_config_29.parquet, Collard_2001_example1_config_3.parquet, Collard_2001_example1_config_30.parquet, Collard_2001_example1_config_31.parquet, Collard_2001_example1_config_32.parquet, Collard_2001_example1_config_33.parquet, Collard_2001_example1_config_34.parquet, Collard_2001_example1_config_35.parquet, Collard_2001_example1_config_36.parquet, Collard_2001_example1_config_37.parquet, Collard_2001_example1_config_38.parquet, Collard_2001_example1_config_39.parquet, Collard_2001_example1_config_4.parquet, Collard_2001_example1_config_5.parquet, Collard_2001_example1_config_6.parquet, Collard_2001_example1_config_7.parquet, Collard_2001_example1_config_8.parquet, Collard_2001_example1_config_9.parquet
```
- **state_names**: `6` items

```
y, c, k, a, h, b
```
- **action_names**: `1` items

```
Consumption
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.29500000000000004`
- **mean_pairwise_coverage**: `0.705`
- **mean_episode_embedding_knn**: `0.0`
- **nearest_env**: `Hansen_1985`
- **shared_state_frac**: `0.6666666666666666`
- **intersection_over_union**: `0.36363636363636365`
- **intra_over_inter**: `0.0`

### RBC_news_shock_model

- **n_episodes**: `40`
- **env_names**: `40` items

```
RBC_news_shock_model_config_0.parquet, RBC_news_shock_model_config_1.parquet, RBC_news_shock_model_config_10.parquet, RBC_news_shock_model_config_11.parquet, RBC_news_shock_model_config_12.parquet, RBC_news_shock_model_config_13.parquet, RBC_news_shock_model_config_14.parquet, RBC_news_shock_model_config_15.parquet, RBC_news_shock_model_config_16.parquet, RBC_news_shock_model_config_17.parquet, RBC_news_shock_model_config_18.parquet, RBC_news_shock_model_config_19.parquet, RBC_news_shock_model_config_2.parquet, RBC_news_shock_model_config_20.parquet, RBC_news_shock_model_config_21.parquet, RBC_news_shock_model_config_22.parquet, RBC_news_shock_model_config_23.parquet, RBC_news_shock_model_config_24.parquet, RBC_news_shock_model_config_25.parquet, RBC_news_shock_model_config_26.parquet, RBC_news_shock_model_config_27.parquet, RBC_news_shock_model_config_28.parquet, RBC_news_shock_model_config_29.parquet, RBC_news_shock_model_config_3.parquet, RBC_news_shock_model_config_30.parquet, RBC_news_shock_model_config_31.parquet, RBC_news_shock_model_config_32.parquet, RBC_news_shock_model_config_33.parquet, RBC_news_shock_model_config_34.parquet, RBC_news_shock_model_config_35.parquet, RBC_news_shock_model_config_36.parquet, RBC_news_shock_model_config_37.parquet, RBC_news_shock_model_config_38.parquet, RBC_news_shock_model_config_39.parquet, RBC_news_shock_model_config_4.parquet, RBC_news_shock_model_config_5.parquet, RBC_news_shock_model_config_6.parquet, RBC_news_shock_model_config_7.parquet, RBC_news_shock_model_config_8.parquet, RBC_news_shock_model_config_9.parquet
```
- **state_names**: `8` items

```
{y}, {c}, {k}, {l}, {z}, {r}, {w}, {i}
```
- **action_names**: `1` items

```
Investment
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.29625`
- **mean_pairwise_coverage**: `0.70375`
- **mean_episode_embedding_knn**: `0.0`
- **nearest_env**: `RBC_state_dependent_GIRF`
- **shared_state_frac**: `1.0`
- **intersection_over_union**: `0.8888888888888888`
- **intra_over_inter**: `0.0`

### McCandless_2008_Chapter_9

- **n_episodes**: `40`
- **env_names**: `40` items

```
McCandless_2008_Chapter_9_config_0.parquet, McCandless_2008_Chapter_9_config_1.parquet, McCandless_2008_Chapter_9_config_10.parquet, McCandless_2008_Chapter_9_config_11.parquet, McCandless_2008_Chapter_9_config_12.parquet, McCandless_2008_Chapter_9_config_13.parquet, McCandless_2008_Chapter_9_config_14.parquet, McCandless_2008_Chapter_9_config_15.parquet, McCandless_2008_Chapter_9_config_16.parquet, McCandless_2008_Chapter_9_config_17.parquet, McCandless_2008_Chapter_9_config_18.parquet, McCandless_2008_Chapter_9_config_19.parquet, McCandless_2008_Chapter_9_config_2.parquet, McCandless_2008_Chapter_9_config_20.parquet, McCandless_2008_Chapter_9_config_21.parquet, McCandless_2008_Chapter_9_config_22.parquet, McCandless_2008_Chapter_9_config_23.parquet, McCandless_2008_Chapter_9_config_24.parquet, McCandless_2008_Chapter_9_config_25.parquet, McCandless_2008_Chapter_9_config_26.parquet, McCandless_2008_Chapter_9_config_27.parquet, McCandless_2008_Chapter_9_config_28.parquet, McCandless_2008_Chapter_9_config_29.parquet, McCandless_2008_Chapter_9_config_3.parquet, McCandless_2008_Chapter_9_config_30.parquet, McCandless_2008_Chapter_9_config_31.parquet, McCandless_2008_Chapter_9_config_32.parquet, McCandless_2008_Chapter_9_config_33.parquet, McCandless_2008_Chapter_9_config_34.parquet, McCandless_2008_Chapter_9_config_35.parquet, McCandless_2008_Chapter_9_config_36.parquet, McCandless_2008_Chapter_9_config_37.parquet, McCandless_2008_Chapter_9_config_38.parquet, McCandless_2008_Chapter_9_config_39.parquet, McCandless_2008_Chapter_9_config_4.parquet, McCandless_2008_Chapter_9_config_5.parquet, McCandless_2008_Chapter_9_config_6.parquet, McCandless_2008_Chapter_9_config_7.parquet, McCandless_2008_Chapter_9_config_8.parquet, McCandless_2008_Chapter_9_config_9.parquet
```
- **state_names**: `10` items

```
W, r, C, K, H, M, P, g, \lambda, y
```
- **action_names**: `1` items

```
Real Consumption
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.366`
- **mean_pairwise_coverage**: `0.634`
- **mean_episode_embedding_knn**: `0.0`
- **nearest_env**: `McCandless_2008_Chapter_13`
- **shared_state_frac**: `0.9`
- **intersection_over_union**: `0.6`
- **intra_over_inter**: `0.0`

### FV_et_al_2007_ABCD

- **n_episodes**: `40`
- **env_names**: `40` items

```
FV_et_al_2007_ABCD_config_0.parquet, FV_et_al_2007_ABCD_config_1.parquet, FV_et_al_2007_ABCD_config_10.parquet, FV_et_al_2007_ABCD_config_11.parquet, FV_et_al_2007_ABCD_config_12.parquet, FV_et_al_2007_ABCD_config_13.parquet, FV_et_al_2007_ABCD_config_14.parquet, FV_et_al_2007_ABCD_config_15.parquet, FV_et_al_2007_ABCD_config_16.parquet, FV_et_al_2007_ABCD_config_17.parquet, FV_et_al_2007_ABCD_config_18.parquet, FV_et_al_2007_ABCD_config_19.parquet, FV_et_al_2007_ABCD_config_2.parquet, FV_et_al_2007_ABCD_config_20.parquet, FV_et_al_2007_ABCD_config_21.parquet, FV_et_al_2007_ABCD_config_22.parquet, FV_et_al_2007_ABCD_config_23.parquet, FV_et_al_2007_ABCD_config_24.parquet, FV_et_al_2007_ABCD_config_25.parquet, FV_et_al_2007_ABCD_config_26.parquet, FV_et_al_2007_ABCD_config_27.parquet, FV_et_al_2007_ABCD_config_28.parquet, FV_et_al_2007_ABCD_config_29.parquet, FV_et_al_2007_ABCD_config_3.parquet, FV_et_al_2007_ABCD_config_30.parquet, FV_et_al_2007_ABCD_config_31.parquet, FV_et_al_2007_ABCD_config_32.parquet, FV_et_al_2007_ABCD_config_33.parquet, FV_et_al_2007_ABCD_config_34.parquet, FV_et_al_2007_ABCD_config_35.parquet, FV_et_al_2007_ABCD_config_36.parquet, FV_et_al_2007_ABCD_config_37.parquet, FV_et_al_2007_ABCD_config_38.parquet, FV_et_al_2007_ABCD_config_39.parquet, FV_et_al_2007_ABCD_config_4.parquet, FV_et_al_2007_ABCD_config_5.parquet, FV_et_al_2007_ABCD_config_6.parquet, FV_et_al_2007_ABCD_config_7.parquet, FV_et_al_2007_ABCD_config_8.parquet, FV_et_al_2007_ABCD_config_9.parquet
```
- **state_names**: `3` items

```
y, c, {y - c}
```
- **action_names**: `1` items

```
Consumption
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.3833333333333333`
- **mean_pairwise_coverage**: `0.6166666666666667`
- **mean_episode_embedding_knn**: `0.0`
- **nearest_env**: `Hansen_1985`
- **shared_state_frac**: `0.6666666666666666`
- **intersection_over_union**: `0.2`
- **intra_over_inter**: `0.0`

### Aguiar_Gopinath_2007

- **n_episodes**: `10`
- **env_names**: `10` items

```
Aguiar_Gopinath_2007_config_0.parquet, Aguiar_Gopinath_2007_config_1.parquet, Aguiar_Gopinath_2007_config_2.parquet, Aguiar_Gopinath_2007_config_3.parquet, Aguiar_Gopinath_2007_config_4.parquet, Aguiar_Gopinath_2007_config_5.parquet, Aguiar_Gopinath_2007_config_6.parquet, Aguiar_Gopinath_2007_config_7.parquet, Aguiar_Gopinath_2007_config_8.parquet, Aguiar_Gopinath_2007_config_9.parquet
```
- **state_names**: `4` items

```
{K}, {D}, {A}, {G}
```
- **action_names**: `2` items

```
Consumption, Labor
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.6312499999999999`
- **mean_pairwise_coverage**: `0.36875000000000013`
- **mean_episode_embedding_knn**: `0.0`
- **nearest_env**: `Gali_2008_chapter_2`
- **shared_state_frac**: `0.25`
- **intersection_over_union**: `0.08333333333333333`
- **intra_over_inter**: `0.0`

### Gali_2008_chapter_2

- **n_episodes**: `40`
- **env_names**: `40` items

```
Gali_2008_chapter_2_config_0.parquet, Gali_2008_chapter_2_config_1.parquet, Gali_2008_chapter_2_config_10.parquet, Gali_2008_chapter_2_config_11.parquet, Gali_2008_chapter_2_config_12.parquet, Gali_2008_chapter_2_config_13.parquet, Gali_2008_chapter_2_config_14.parquet, Gali_2008_chapter_2_config_15.parquet, Gali_2008_chapter_2_config_16.parquet, Gali_2008_chapter_2_config_17.parquet, Gali_2008_chapter_2_config_18.parquet, Gali_2008_chapter_2_config_19.parquet, Gali_2008_chapter_2_config_2.parquet, Gali_2008_chapter_2_config_20.parquet, Gali_2008_chapter_2_config_21.parquet, Gali_2008_chapter_2_config_22.parquet, Gali_2008_chapter_2_config_23.parquet, Gali_2008_chapter_2_config_24.parquet, Gali_2008_chapter_2_config_25.parquet, Gali_2008_chapter_2_config_26.parquet, Gali_2008_chapter_2_config_27.parquet, Gali_2008_chapter_2_config_28.parquet, Gali_2008_chapter_2_config_29.parquet, Gali_2008_chapter_2_config_3.parquet, Gali_2008_chapter_2_config_30.parquet, Gali_2008_chapter_2_config_31.parquet, Gali_2008_chapter_2_config_32.parquet, Gali_2008_chapter_2_config_33.parquet, Gali_2008_chapter_2_config_34.parquet, Gali_2008_chapter_2_config_35.parquet, Gali_2008_chapter_2_config_36.parquet, Gali_2008_chapter_2_config_37.parquet, Gali_2008_chapter_2_config_38.parquet, Gali_2008_chapter_2_config_39.parquet, Gali_2008_chapter_2_config_4.parquet, Gali_2008_chapter_2_config_5.parquet, Gali_2008_chapter_2_config_6.parquet, Gali_2008_chapter_2_config_7.parquet, Gali_2008_chapter_2_config_8.parquet, Gali_2008_chapter_2_config_9.parquet
```
- **state_names**: `9` items

```
{C}, {\frac{W}{P}}, {\Pi}, {A}, {N}, {R^n}, {R^{r}}, {Y}, {\Delta M}
```
- **action_names**: `1` items

```
Nominal Interest Rate
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.6416666666666667`
- **mean_pairwise_coverage**: `0.3583333333333333`
- **mean_episode_embedding_knn**: `0.0`
- **nearest_env**: `Aguiar_Gopinath_2007`
- **shared_state_frac**: `0.1111111111111111`
- **intersection_over_union**: `0.08333333333333333`
- **intra_over_inter**: `0.0`

### SGU_2004

- **n_episodes**: `40`
- **env_names**: `40` items

```
SGU_2004_config_0.parquet, SGU_2004_config_1.parquet, SGU_2004_config_10.parquet, SGU_2004_config_11.parquet, SGU_2004_config_12.parquet, SGU_2004_config_13.parquet, SGU_2004_config_14.parquet, SGU_2004_config_15.parquet, SGU_2004_config_16.parquet, SGU_2004_config_17.parquet, SGU_2004_config_18.parquet, SGU_2004_config_19.parquet, SGU_2004_config_2.parquet, SGU_2004_config_20.parquet, SGU_2004_config_21.parquet, SGU_2004_config_22.parquet, SGU_2004_config_23.parquet, SGU_2004_config_24.parquet, SGU_2004_config_25.parquet, SGU_2004_config_26.parquet, SGU_2004_config_27.parquet, SGU_2004_config_28.parquet, SGU_2004_config_29.parquet, SGU_2004_config_3.parquet, SGU_2004_config_30.parquet, SGU_2004_config_31.parquet, SGU_2004_config_32.parquet, SGU_2004_config_33.parquet, SGU_2004_config_34.parquet, SGU_2004_config_35.parquet, SGU_2004_config_36.parquet, SGU_2004_config_37.parquet, SGU_2004_config_38.parquet, SGU_2004_config_39.parquet, SGU_2004_config_4.parquet, SGU_2004_config_5.parquet, SGU_2004_config_6.parquet, SGU_2004_config_7.parquet, SGU_2004_config_8.parquet, SGU_2004_config_9.parquet
```
- **state_names**: `3` items

```
{c}, {k}, {a}
```
- **action_names**: `1` items

```
Capital
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.7166666666666667`
- **mean_pairwise_coverage**: `0.2833333333333333`
- **mean_episode_embedding_knn**: `0.0`
- **nearest_env**: `RBC_news_shock_model`
- **shared_state_frac**: `0.6666666666666666`
- **intersection_over_union**: `0.2222222222222222`
- **intra_over_inter**: `0.0`

### GarciaCicco_et_al_2010

- **n_episodes**: `40`
- **env_names**: `40` items

```
GarciaCicco_et_al_2010_config_0.parquet, GarciaCicco_et_al_2010_config_1.parquet, GarciaCicco_et_al_2010_config_10.parquet, GarciaCicco_et_al_2010_config_11.parquet, GarciaCicco_et_al_2010_config_12.parquet, GarciaCicco_et_al_2010_config_13.parquet, GarciaCicco_et_al_2010_config_14.parquet, GarciaCicco_et_al_2010_config_15.parquet, GarciaCicco_et_al_2010_config_16.parquet, GarciaCicco_et_al_2010_config_17.parquet, GarciaCicco_et_al_2010_config_18.parquet, GarciaCicco_et_al_2010_config_19.parquet, GarciaCicco_et_al_2010_config_2.parquet, GarciaCicco_et_al_2010_config_20.parquet, GarciaCicco_et_al_2010_config_21.parquet, GarciaCicco_et_al_2010_config_22.parquet, GarciaCicco_et_al_2010_config_23.parquet, GarciaCicco_et_al_2010_config_24.parquet, GarciaCicco_et_al_2010_config_25.parquet, GarciaCicco_et_al_2010_config_26.parquet, GarciaCicco_et_al_2010_config_27.parquet, GarciaCicco_et_al_2010_config_28.parquet, GarciaCicco_et_al_2010_config_29.parquet, GarciaCicco_et_al_2010_config_3.parquet, GarciaCicco_et_al_2010_config_30.parquet, GarciaCicco_et_al_2010_config_31.parquet, GarciaCicco_et_al_2010_config_32.parquet, GarciaCicco_et_al_2010_config_33.parquet, GarciaCicco_et_al_2010_config_34.parquet, GarciaCicco_et_al_2010_config_35.parquet, GarciaCicco_et_al_2010_config_36.parquet, GarciaCicco_et_al_2010_config_37.parquet, GarciaCicco_et_al_2010_config_38.parquet, GarciaCicco_et_al_2010_config_39.parquet, GarciaCicco_et_al_2010_config_4.parquet, GarciaCicco_et_al_2010_config_5.parquet, GarciaCicco_et_al_2010_config_6.parquet, GarciaCicco_et_al_2010_config_7.parquet, GarciaCicco_et_al_2010_config_8.parquet, GarciaCicco_et_al_2010_config_9.parquet
```
- **state_names**: `8` items

```
Capital, Debt, Output, InterestRate, LoggedProductivity, PreferenceShock, CountryPremiumShock, TechGrowthRate
```
- **action_names**: `3` items

```
Investment, Consumption, HoursWorked
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.8019270833333334`
- **mean_pairwise_coverage**: `0.1980729166666666`
- **mean_episode_embedding_knn**: `6.215477018223412`
- **nearest_env**: `OLG`
- **shared_state_frac**: `0.375`
- **intersection_over_union**: `0.3`
- **intra_over_inter**: `0.18859120789842743`

### RBC_baseline_stoch

- **n_episodes**: `40`
- **env_names**: `40` items

```
RBC_baseline_stoch_config_0.parquet, RBC_baseline_stoch_config_1.parquet, RBC_baseline_stoch_config_10.parquet, RBC_baseline_stoch_config_11.parquet, RBC_baseline_stoch_config_12.parquet, RBC_baseline_stoch_config_13.parquet, RBC_baseline_stoch_config_14.parquet, RBC_baseline_stoch_config_15.parquet, RBC_baseline_stoch_config_16.parquet, RBC_baseline_stoch_config_17.parquet, RBC_baseline_stoch_config_18.parquet, RBC_baseline_stoch_config_19.parquet, RBC_baseline_stoch_config_2.parquet, RBC_baseline_stoch_config_20.parquet, RBC_baseline_stoch_config_21.parquet, RBC_baseline_stoch_config_22.parquet, RBC_baseline_stoch_config_23.parquet, RBC_baseline_stoch_config_24.parquet, RBC_baseline_stoch_config_25.parquet, RBC_baseline_stoch_config_26.parquet, RBC_baseline_stoch_config_27.parquet, RBC_baseline_stoch_config_28.parquet, RBC_baseline_stoch_config_29.parquet, RBC_baseline_stoch_config_3.parquet, RBC_baseline_stoch_config_30.parquet, RBC_baseline_stoch_config_31.parquet, RBC_baseline_stoch_config_32.parquet, RBC_baseline_stoch_config_33.parquet, RBC_baseline_stoch_config_34.parquet, RBC_baseline_stoch_config_35.parquet, RBC_baseline_stoch_config_36.parquet, RBC_baseline_stoch_config_37.parquet, RBC_baseline_stoch_config_38.parquet, RBC_baseline_stoch_config_39.parquet, RBC_baseline_stoch_config_4.parquet, RBC_baseline_stoch_config_5.parquet, RBC_baseline_stoch_config_6.parquet, RBC_baseline_stoch_config_7.parquet, RBC_baseline_stoch_config_8.parquet, RBC_baseline_stoch_config_9.parquet
```
- **state_names**: `2` items

```
Capital, LoggedProductivity
```
- **action_names**: `1` items

```
Consumption
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.857375`
- **mean_pairwise_coverage**: `0.142625`
- **mean_episode_embedding_knn**: `2.9903260677627497`
- **nearest_env**: `GarciaCicco_et_al_2010`
- **shared_state_frac**: `1.0`
- **intersection_over_union**: `0.25`
- **intra_over_inter**: `0.19024272007163584`

### Ramsey_cara

- **n_episodes**: `40`
- **env_names**: `40` items

```
Ramsey_cara_config_0.parquet, Ramsey_cara_config_1.parquet, Ramsey_cara_config_10.parquet, Ramsey_cara_config_11.parquet, Ramsey_cara_config_12.parquet, Ramsey_cara_config_13.parquet, Ramsey_cara_config_14.parquet, Ramsey_cara_config_15.parquet, Ramsey_cara_config_16.parquet, Ramsey_cara_config_17.parquet, Ramsey_cara_config_18.parquet, Ramsey_cara_config_19.parquet, Ramsey_cara_config_2.parquet, Ramsey_cara_config_20.parquet, Ramsey_cara_config_21.parquet, Ramsey_cara_config_22.parquet, Ramsey_cara_config_23.parquet, Ramsey_cara_config_24.parquet, Ramsey_cara_config_25.parquet, Ramsey_cara_config_26.parquet, Ramsey_cara_config_27.parquet, Ramsey_cara_config_28.parquet, Ramsey_cara_config_29.parquet, Ramsey_cara_config_3.parquet, Ramsey_cara_config_30.parquet, Ramsey_cara_config_31.parquet, Ramsey_cara_config_32.parquet, Ramsey_cara_config_33.parquet, Ramsey_cara_config_34.parquet, Ramsey_cara_config_35.parquet, Ramsey_cara_config_36.parquet, Ramsey_cara_config_37.parquet, Ramsey_cara_config_38.parquet, Ramsey_cara_config_39.parquet, Ramsey_cara_config_4.parquet, Ramsey_cara_config_5.parquet, Ramsey_cara_config_6.parquet, Ramsey_cara_config_7.parquet, Ramsey_cara_config_8.parquet, Ramsey_cara_config_9.parquet
```
- **state_names**: `3` items

```
CapitalPerCapita, OutputPerCapita, Labor
```
- **action_names**: `1` items

```
ConsumptionPerCapita
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.8876666666666665`
- **mean_pairwise_coverage**: `0.11233333333333351`
- **mean_episode_embedding_knn**: `3.6036349125529945`
- **nearest_env**: `Ramsey_crra`
- **shared_state_frac**: `1.0`
- **intersection_over_union**: `1.0`
- **intra_over_inter**: `0.8668002765110735`

### Ramsey_upgrade

- **n_episodes**: `40`
- **env_names**: `40` items

```
Ramsey_upgrade_config_0.parquet, Ramsey_upgrade_config_1.parquet, Ramsey_upgrade_config_10.parquet, Ramsey_upgrade_config_11.parquet, Ramsey_upgrade_config_12.parquet, Ramsey_upgrade_config_13.parquet, Ramsey_upgrade_config_14.parquet, Ramsey_upgrade_config_15.parquet, Ramsey_upgrade_config_16.parquet, Ramsey_upgrade_config_17.parquet, Ramsey_upgrade_config_18.parquet, Ramsey_upgrade_config_19.parquet, Ramsey_upgrade_config_2.parquet, Ramsey_upgrade_config_20.parquet, Ramsey_upgrade_config_21.parquet, Ramsey_upgrade_config_22.parquet, Ramsey_upgrade_config_23.parquet, Ramsey_upgrade_config_24.parquet, Ramsey_upgrade_config_25.parquet, Ramsey_upgrade_config_26.parquet, Ramsey_upgrade_config_27.parquet, Ramsey_upgrade_config_28.parquet, Ramsey_upgrade_config_29.parquet, Ramsey_upgrade_config_3.parquet, Ramsey_upgrade_config_30.parquet, Ramsey_upgrade_config_31.parquet, Ramsey_upgrade_config_32.parquet, Ramsey_upgrade_config_33.parquet, Ramsey_upgrade_config_34.parquet, Ramsey_upgrade_config_35.parquet, Ramsey_upgrade_config_36.parquet, Ramsey_upgrade_config_37.parquet, Ramsey_upgrade_config_38.parquet, Ramsey_upgrade_config_39.parquet, Ramsey_upgrade_config_4.parquet, Ramsey_upgrade_config_5.parquet, Ramsey_upgrade_config_6.parquet, Ramsey_upgrade_config_7.parquet, Ramsey_upgrade_config_8.parquet, Ramsey_upgrade_config_9.parquet
```
- **state_names**: `4` items

```
CapitalPerEffectiveLabor, LoggedProductivity, Labor, InterestRate
```
- **action_names**: `1` items

```
ConsumptionPerEffectiveLabor
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.890625`
- **mean_pairwise_coverage**: `0.109375`
- **mean_episode_embedding_knn**: `5.299059017840868`
- **nearest_env**: `GarciaCicco_et_al_2010`
- **shared_state_frac**: `0.5`
- **intersection_over_union**: `0.2`
- **intra_over_inter**: `0.155530532044845`

### Ramsey_crra

- **n_episodes**: `40`
- **env_names**: `40` items

```
Ramsey_crra_config_0.parquet, Ramsey_crra_config_1.parquet, Ramsey_crra_config_10.parquet, Ramsey_crra_config_11.parquet, Ramsey_crra_config_12.parquet, Ramsey_crra_config_13.parquet, Ramsey_crra_config_14.parquet, Ramsey_crra_config_15.parquet, Ramsey_crra_config_16.parquet, Ramsey_crra_config_17.parquet, Ramsey_crra_config_18.parquet, Ramsey_crra_config_19.parquet, Ramsey_crra_config_2.parquet, Ramsey_crra_config_20.parquet, Ramsey_crra_config_21.parquet, Ramsey_crra_config_22.parquet, Ramsey_crra_config_23.parquet, Ramsey_crra_config_24.parquet, Ramsey_crra_config_25.parquet, Ramsey_crra_config_26.parquet, Ramsey_crra_config_27.parquet, Ramsey_crra_config_28.parquet, Ramsey_crra_config_29.parquet, Ramsey_crra_config_3.parquet, Ramsey_crra_config_30.parquet, Ramsey_crra_config_31.parquet, Ramsey_crra_config_32.parquet, Ramsey_crra_config_33.parquet, Ramsey_crra_config_34.parquet, Ramsey_crra_config_35.parquet, Ramsey_crra_config_36.parquet, Ramsey_crra_config_37.parquet, Ramsey_crra_config_38.parquet, Ramsey_crra_config_39.parquet, Ramsey_crra_config_4.parquet, Ramsey_crra_config_5.parquet, Ramsey_crra_config_6.parquet, Ramsey_crra_config_7.parquet, Ramsey_crra_config_8.parquet, Ramsey_crra_config_9.parquet
```
- **state_names**: `3` items

```
CapitalPerCapita, OutputPerCapita, Labor
```
- **action_names**: `1` items

```
ConsumptionPerCapita
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.8914166666666666`
- **mean_pairwise_coverage**: `0.10858333333333337`
- **mean_episode_embedding_knn**: `3.492696798214516`
- **nearest_env**: `Ramsey_cara`
- **shared_state_frac**: `1.0`
- **intersection_over_union**: `1.0`
- **intra_over_inter**: `0.8668002765110733`

### Ramsey_base

- **n_episodes**: `40`
- **env_names**: `40` items

```
Ramsey_base_config_0.parquet, Ramsey_base_config_1.parquet, Ramsey_base_config_10.parquet, Ramsey_base_config_11.parquet, Ramsey_base_config_12.parquet, Ramsey_base_config_13.parquet, Ramsey_base_config_14.parquet, Ramsey_base_config_15.parquet, Ramsey_base_config_16.parquet, Ramsey_base_config_17.parquet, Ramsey_base_config_18.parquet, Ramsey_base_config_19.parquet, Ramsey_base_config_2.parquet, Ramsey_base_config_20.parquet, Ramsey_base_config_21.parquet, Ramsey_base_config_22.parquet, Ramsey_base_config_23.parquet, Ramsey_base_config_24.parquet, Ramsey_base_config_25.parquet, Ramsey_base_config_26.parquet, Ramsey_base_config_27.parquet, Ramsey_base_config_28.parquet, Ramsey_base_config_29.parquet, Ramsey_base_config_3.parquet, Ramsey_base_config_30.parquet, Ramsey_base_config_31.parquet, Ramsey_base_config_32.parquet, Ramsey_base_config_33.parquet, Ramsey_base_config_34.parquet, Ramsey_base_config_35.parquet, Ramsey_base_config_36.parquet, Ramsey_base_config_37.parquet, Ramsey_base_config_38.parquet, Ramsey_base_config_39.parquet, Ramsey_base_config_4.parquet, Ramsey_base_config_5.parquet, Ramsey_base_config_6.parquet, Ramsey_base_config_7.parquet, Ramsey_base_config_8.parquet, Ramsey_base_config_9.parquet
```
- **state_names**: `2` items

```
Capital, Output
```
- **action_names**: `1` items

```
Consumption
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.912125`
- **mean_pairwise_coverage**: `0.08787500000000004`
- **mean_episode_embedding_knn**: `2.156283033253907`
- **nearest_env**: `OLG`
- **shared_state_frac**: `1.0`
- **intersection_over_union**: `0.4`
- **intra_over_inter**: `0.18857620516969123`

### RBC_baseline_pf

- **n_episodes**: `40`
- **env_names**: `40` items

```
RBC_baseline_pf_config_0.parquet, RBC_baseline_pf_config_1.parquet, RBC_baseline_pf_config_10.parquet, RBC_baseline_pf_config_11.parquet, RBC_baseline_pf_config_12.parquet, RBC_baseline_pf_config_13.parquet, RBC_baseline_pf_config_14.parquet, RBC_baseline_pf_config_15.parquet, RBC_baseline_pf_config_16.parquet, RBC_baseline_pf_config_17.parquet, RBC_baseline_pf_config_18.parquet, RBC_baseline_pf_config_19.parquet, RBC_baseline_pf_config_2.parquet, RBC_baseline_pf_config_20.parquet, RBC_baseline_pf_config_21.parquet, RBC_baseline_pf_config_22.parquet, RBC_baseline_pf_config_23.parquet, RBC_baseline_pf_config_24.parquet, RBC_baseline_pf_config_25.parquet, RBC_baseline_pf_config_26.parquet, RBC_baseline_pf_config_27.parquet, RBC_baseline_pf_config_28.parquet, RBC_baseline_pf_config_29.parquet, RBC_baseline_pf_config_3.parquet, RBC_baseline_pf_config_30.parquet, RBC_baseline_pf_config_31.parquet, RBC_baseline_pf_config_32.parquet, RBC_baseline_pf_config_33.parquet, RBC_baseline_pf_config_34.parquet, RBC_baseline_pf_config_35.parquet, RBC_baseline_pf_config_36.parquet, RBC_baseline_pf_config_37.parquet, RBC_baseline_pf_config_38.parquet, RBC_baseline_pf_config_39.parquet, RBC_baseline_pf_config_4.parquet, RBC_baseline_pf_config_5.parquet, RBC_baseline_pf_config_6.parquet, RBC_baseline_pf_config_7.parquet, RBC_baseline_pf_config_8.parquet, RBC_baseline_pf_config_9.parquet
```
- **state_names**: `2` items

```
Capital, LoggedProductivity
```
- **action_names**: `1` items

```
Consumption
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.956861111111111`
- **mean_pairwise_coverage**: `0.04313888888888895`
- **mean_episode_embedding_knn**: `3.754104340355299`
- **nearest_env**: `RBC_baseline_stoch`
- **shared_state_frac**: `1.0`
- **intersection_over_union**: `1.0`
- **intra_over_inter**: `0.42434634375861124`

### OLG

- **n_episodes**: `40`
- **env_names**: `40` items

```
OLG_config_0.parquet, OLG_config_1.parquet, OLG_config_10.parquet, OLG_config_11.parquet, OLG_config_12.parquet, OLG_config_13.parquet, OLG_config_14.parquet, OLG_config_15.parquet, OLG_config_16.parquet, OLG_config_17.parquet, OLG_config_18.parquet, OLG_config_19.parquet, OLG_config_2.parquet, OLG_config_20.parquet, OLG_config_21.parquet, OLG_config_22.parquet, OLG_config_23.parquet, OLG_config_24.parquet, OLG_config_25.parquet, OLG_config_26.parquet, OLG_config_27.parquet, OLG_config_28.parquet, OLG_config_29.parquet, OLG_config_3.parquet, OLG_config_30.parquet, OLG_config_31.parquet, OLG_config_32.parquet, OLG_config_33.parquet, OLG_config_34.parquet, OLG_config_35.parquet, OLG_config_36.parquet, OLG_config_37.parquet, OLG_config_38.parquet, OLG_config_39.parquet, OLG_config_4.parquet, OLG_config_5.parquet, OLG_config_6.parquet, OLG_config_7.parquet, OLG_config_8.parquet, OLG_config_9.parquet
```
- **state_names**: `5` items

```
Capital, Output, Wage, InterestRate, Savings
```
- **action_names**: `1` items

```
Savings
```
- **endogenous_names**: `0` items

```

```
- **mean_pairwise_vacancy**: `0.95695`
- **mean_pairwise_coverage**: `0.04305000000000003`
- **mean_episode_embedding_knn**: `3.9131027062749815`
- **nearest_env**: `GarciaCicco_et_al_2010`
- **shared_state_frac**: `0.6`
- **intersection_over_union**: `0.3`
- **intra_over_inter**: `0.18859120789842743`

