INFO = '''
---
  file: classifier.pkl
  type: LogisticRegression
  n_features_in: 25
  classes: ['hyper', 'hypo', 'normal']
---
  file: global_feature_scaler.pkl
  type: StandardScaler
  n_features_in: 25
---
  file: glucose_regressor.pkl
  type: RANSACRegressor
  n_features_in: 23
---
  file: hb_regressor.pkl
  type: RANSACRegressor
  n_features_in: 23
---
  file: hyper_models.pkl
  type: dict
  keys: ['sbp_model', 'dbp_model']
  key_sbp_model_type: RandomForestRegressor
  key_sbp_model_n_features: 25
  key_dbp_model_type: RandomForestRegressor
  key_dbp_model_n_features: 25
---
  file: hypo_models.pkl
  type: dict
  keys: ['sbp_model', 'dbp_model']
  key_sbp_model_type: RandomForestRegressor
  key_sbp_model_n_features: 25
  key_dbp_model_type: RandomForestRegressor
  key_dbp_model_n_features: 25
---
  file: normal_models.pkl
  type: dict
  keys: ['sbp_model', 'dbp_model']
  key_sbp_model_type: RandomForestRegressor
  key_sbp_model_n_features: 25
  key_dbp_model_type: RandomForestRegressor
  key_dbp_model_n_features: 25
---
  file: scaler_glucose.pkl
  type: StandardScaler
  n_features_in: 23
---
  file: scaler_hb.pkl
  type: StandardScaler
  n_features_in: 23
'''