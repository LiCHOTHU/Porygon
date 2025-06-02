import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "real_data"
methods = ["rgb", "dp3", "3dda", "adapt3r"]
conditions = ["mt", "cam_change"]

# Collect average completions
avg_completions = {method: {} for method in methods}

for method in methods:
    for condition in conditions:
        file_path = os.path.join(base_dir, method, f"{condition}.csv")
        df = pd.read_csv(file_path, index_col=0)

        # Drop rows that are entirely empty
        df.dropna(how="all", inplace=True)

        # Handle the case where the last row is the aggregate row
        try:
            avg_completion = float(df.iloc[-1]["Average Completion"])
        except Exception:
            # If the final row is not numeric, compute the mean manually
            avg_completion = df["Average Completion"].astype(float).mean()
        
        avg_completions[method][condition] = avg_completion

# # Plotting
# labels = methods
# mt_vals = [avg_completions[m]["mt"] for m in methods]
# cam_vals = [avg_completions[m]["cam_change"] for m in methods]

# x = range(len(methods))
# width = 0.35

# fig, ax = plt.subplots(figsize=(8, 5))
# ax.bar([i - width/2 - 0.01 for i in x], mt_vals, width, label='MT', color='blue')
# ax.bar([i + width/2 + 0.01 for i in x], cam_vals, width, label='Cam Change', color='blue')

# ax.set_ylabel('Average Completion')
# ax.set_title('Average Completion by Method and Condition')
# ax.set_xticks(x)
# ax.set_xticklabels([m.upper() for m in methods])
# ax.legend()

# plt.tight_layout()
# plt.show()

# Plotting
labels = methods
mt_vals = [avg_completions[m]["mt"] for m in methods]
cam_vals = [avg_completions[m]["cam_change"] for m in methods]

x = range(len(methods))
mt_width = 0.35
cam_width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

# MT bars (lighter and slightly offset left)
ax.bar([i - mt_width/2 - 0.01 for i in x], mt_vals, mt_width, label='MT',
       color='lightskyblue', edgecolor='black', linewidth=1.)

# Cam Change bars (emphasized)
bars = ax.bar([i + cam_width/2 + 0.01 for i in x], cam_vals, cam_width, label='Cam Change',
              color='steelblue', edgecolor='black', linewidth=1.)

# Add annotations on top of cam change bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # Offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Average Completion')
ax.set_title('Average Completion by Method — Emphasis on Cam Change')
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in methods])
ax.legend()

plt.tight_layout()
plt.show()
