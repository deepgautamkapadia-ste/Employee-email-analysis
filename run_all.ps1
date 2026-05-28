$ErrorActionPreference = "Stop"

Write-Host "Running Task 1: Sentiment labeling"
python task1_sentiment_labeling.py --provider local --resume --checkpoint-every 50

Write-Host "Running Task 2: EDA"
python task2_eda.py

Write-Host "Running Task 3: Monthly score calculation"
python task3_score_calculation.py

Write-Host "Running Task 4: Employee ranking"
python task4_employee_ranking.py

Write-Host "Running Task 5: Flight risk detection"
python task5_flight_risk.py

Write-Host "Running Task 6: Predictive modeling"
python task6_predictive_modeling.py

Write-Host "Pipeline complete."
