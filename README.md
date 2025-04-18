# Normalized Triangular MFs
!pip install -q scikit-fuzzy
import numpy as np
import pandas as pd
import skfuzzy as fuzz #Fixed typo here

# Original M1 values
M1 = np.array([242, 274, 366, 272, 260, 290, 378, 276, 402, 406, 526, 340, 316, 388, 386])

# Normalize M1 for consistency (optional but recommended for fuzzification)
M1_norm = (M1 - M1.min()) / (M1.max() - M1.min())

# Calculate min, mean, max of normalized M1
x_min = M1_norm.min()
x_max = M1_norm.max()
x_mid = M1_norm.mean()

# Compute fuzzy memberships using triangular functions
low_mf = fuzz.trimf(M1_norm, [x_min, x_min, x_mid])
med_mf = fuzz.trimf(M1_norm, [x_min, x_mid, x_max])
high_mf = fuzz.trimf(M1_norm, [x_mid, x_max, x_max])

# Prepare DataFrame for output
df = pd.DataFrame({
    'M1_Original': M1,
    'M1_Normalized': M1_norm.round(4),
    'Low_Membership': low_mf.round(4),
    'Medium_Membership': med_mf.round(4),
    'High_Membership': high_mf.round(4)
})

# Display the table
print(df.to_string(index=False))
