To run the experiments in order, run "run_full_experiments.py".


| Script | Description|
| -------- | -------- | 
| zt_dataset_generator.py| Generates the dataset with a configurable number of users, sessions, period, attack campaigns, attack - benign ratio|
|validate_dataset.py| Validates the dataset to check for conformance to data rules|
|dataset_realism_analysis.py| Validates the dataset and shows the attack distribution based on the attributes|
|inspect_context_signals.py|  Verify the distribution of contextual signals|
|compute_risk_weights.py|  Generate the weights (feature importance) of the risk signals  |
|evaluate_risk_model.py|  Evaluate performance metrics of models|
|compute_decision_thresholds.py|  Generate step-up and block decision thresholds from the dataset|
|evaluate_policy.py|  Study of outcomes from policy decisions|
|user_friction_analysis.py|  Study how many benign users are challenged or blocked |
|attack_detection_delay.py|  Study how soon an attack can be detected by the proposed framework|
|ablation_study.py|  Remove risk signals and signal groups, and study the response of the proposed framework|
|attack_intensity_experiment.py|  Modify the attack intensity and study the response of the proposed framework|
|compare_performance_with_baselines.py|  Compare the proposed framework with baseline models and works from references|
|generate_results_table.py|  Generate results tables from the findings|
|generate_all_figures.py|  Generate all figures from the findings|


