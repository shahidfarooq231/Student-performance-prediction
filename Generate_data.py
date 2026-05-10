import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1000

data = pd.DataFrame({
    "study_hours": np.random.randint(0, 12, rows),
    "attendance": np.random.randint(40, 100, rows),
    "previous_marks": np.random.randint(30, 100, rows),
    "sleep_hours": np.random.randint(4, 10, rows),
    "assignments": np.random.randint(0, 10, rows),
})

score = (
    0.3 * data["study_hours"] +
    0.2 * data["attendance"] / 10 +
    0.4 * data["previous_marks"] / 10 +
    0.1 * data["assignments"] +
    0.1 * data["sleep_hours"]
)

noise = np.random.normal(0, 0.8, rows)

final_score = score + noise

data["result"] = (final_score > 6.5).astype(int)

data.to_csv("student_data.csv", index=False)

print("Dataset created successfully!")
