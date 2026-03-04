"""
Demo dataset generator for mlcli tabular workflow.
Generates a realistic customer churn prediction dataset.

Usage:
    python generate_demo_data.py

Output:
    data/raw/customers.csv
"""

import random
import os

random.seed(42)

os.makedirs("data/raw", exist_ok=True)

header = "age,income,tenure_months,num_products,has_credit_card,is_active,country,education,churn"

rows = []
countries = ["India", "USA", "UK", "Germany", "Australia"]
educations = ["HS", "Bachelors", "Masters", "PhD"]

for i in range(200):
    age = random.randint(22, 65)
    income = random.randint(20000, 150000)
    tenure = random.randint(1, 120)
    num_products = random.randint(1, 4)
    has_cc = random.choice([0, 1])
    is_active = random.choice([0, 1])
    country = random.choice(countries)
    education = random.choice(educations)

    # Churn logic: more likely if low tenure, inactive, few products
    churn_score = 0
    if tenure < 12:
        churn_score += 2
    if is_active == 0:
        churn_score += 2
    if num_products == 1:
        churn_score += 1
    if income < 40000:
        churn_score += 1
    churn = 1 if churn_score >= 3 else 0

    rows.append(f"{age},{income},{tenure},{num_products},{has_cc},{is_active},{country},{education},{churn}")

with open("data/raw/customers.csv", "w") as f:
    f.write(header + "\n")
    f.write("\n".join(rows))

print(f"Created data/raw/customers.csv ({len(rows)} rows)")
print("Columns: age, income, tenure_months, num_products, has_credit_card, is_active, country, education")
print("Target:  churn (0=stayed, 1=churned)")
print()
print("Next steps:")
print("  mlcli preprocess -i data/raw/customers.csv -t churn -o data/processed")
