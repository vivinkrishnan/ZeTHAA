To run the experiments in order, run "run_full_experiments.py".

"zt_dataset_generator.py" - generates the dataset with a configurable number of users, sessions, period, attack campaigns, attack - benign ratio
"validate_dataset.py" - validates the dataset to check for conformance to data rules
"dataset_realism_analysis.py" - validates the dataset and shows the attack distribution based on the attributes
"inspect_context_signals.py" - verify the distribution of contextual signals
"compute_risk_weights.py" - Generate the weights (feature importance) of the risk signals  
"evaluate_risk_model.py" - Evaluate performance metrics of models
"compute_decision_thresholds.py" - generate step-up and block decision thresholds from the dataset
"evaluate_policy.py" - study of outcomes from policy decisions
"user_friction_analysis.py" - study how many benign users are challenged or blocked 
"attack_detection_delay.py" - study how soon an attack can be detected by the proposed framework
"ablation_study.py" - remove risk signals and signal groups, and study the response of the proposed framework
"attack_intensity_experiment.py" - modify the attack intensity and study the response of the proposed framework
"compare_performance_with_baselines.py" - compare the proposed framework with baseline models and works from references
"generate_results_table.py" - generate results tables from the findings
"generate_all_figures.py" - generate all figures from the findings
