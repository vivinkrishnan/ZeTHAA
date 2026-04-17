import subprocess

scripts=[
"zt_dataset_generator.py",
"validate_dataset.py",
"dataset_realism_analysis.py",
"inspect_context_signals.py",
"compute_risk_weights.py",
"evaluate_risk_model.py",
"compute_decision_thresholds.py",
"evaluate_policy.py",
"user_friction_analysis.py",
"attack_detection_delay.py",
"statistical_significance_test.py",
"ablation_study.py",
"attack_intensity_experiment.py",
"compare_performance_with_baselines.py",
"generate_results_table.py",
"generate_all_figures.py"
]

for s in scripts:
    subprocess.run(["python",s],check=True)

print("Experiment pipeline completed")