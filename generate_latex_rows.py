import pandas as pd
import re
import textwrap

# Load CSV
df = pd.read_csv('updated_interpretation_table_v2.csv')

# Add ID and round Difference
df.insert(0, 'ID', range(1, len(df) + 1))
df['Difference'] = (df['% Zeros (With)'] - df['% Zeros (Without)']).round(1)

# Escape underscores in 'Test'
df['Test'] = df['Test'].apply(lambda x: x.replace('_', r' '))
df['Test'] = df['Test'].apply(lambda x: x.replace('Smell type: ', r''))
df['Test'] = df['Test'].apply(lambda x: x.replace('Subtype: ', r''))
df['Interpretation'] = df['Interpretation'].apply(lambda x: x.replace('%', r'\%'))
df['Interpretation'] = df['Interpretation'].apply(lambda x: x.replace('_', r' '))


# Abbreviate Metric column
df['Metric'] = df['Metric'].replace({'Completeness': 'Comp.', 'Correctness': 'Corr.'})

# Round Cramer's V to 3 decimal places, handle zeros
def round_cramers_v(val):
    try:
        return f"{round(float(val), 3):.3f}"
    except:
        print('Smth went wrong with rounding!!')
        return val  # If something goes wrong, return original
    
# Round p-value appropriately
def round_p_value(val):
    try:
        val_str = str(val)
        if 'e' in val_str or 'E' in val_str:
            match = re.match(r'([0-9.]+)[eE]([-+]?\d+)', val_str)
            if match:
                base = float(match.group(1))
                exponent = match.group(2)
                rounded_base = round(base, 4)
                return f"{rounded_base}e{exponent}"
            else:
                return val_str
        else:
            return f"{round(float(val), 4):.4f}"
    except:
        return val  # In case of any issue, return original
    
df["Cramer's V"] = df["Cramer's V"].apply(round_cramers_v)
df['p-value'] = df['p-value'].apply(round_p_value)

# First table columns
first_table_columns = ['ID', 'Test', 'Metric', 'p-value', 'Significant', "Cramer's V", 'Effect Size']
# Second table columns
second_table_columns = ['ID', '% Zeros (Without)', '% Zeros (With)', 'Difference', 'Impact', 'Interpretation']

df_first = df[first_table_columns]
df_second = df[second_table_columns]

# Function to generate plain LaTeX row strings
def generate_rows(df, wrap_interpretation=False):
    rows = []
    for _, row in df.iterrows():
        formatted = []
        for col in df.columns:
            val = row[col]
            if col == 'Interpretation' and wrap_interpretation:
                val = textwrap.fill(str(val), width=40).replace('\n', '\\newline ')
            formatted.append(str(val))
        rows.append(' & '.join(formatted) + ' \\\\')
    return rows

# Generate LaTeX rows
first_table_rows = generate_rows(df_first)
second_table_rows = generate_rows(df_second, wrap_interpretation=True)

# Print LaTeX row content
print("% --- First Table Rows ---")
for row in first_table_rows:
    print(row)

print("\n% --- Second Table Rows ---")
for row in second_table_rows:
    print(row)
