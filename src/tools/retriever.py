import pandas as pd

pd.set_option('display.max_columns', None)  # Show all columns in the output
pd.set_option('display.max_rows', None)     # Show all rows in the output
pd.set_option('display.width', None)

df = pd.read_csv(r"C:\Users\bekal\Downloads\Minecraft.csv")

print(df.shape)
print(df.iloc[100])
print(df.columns)
print(df['After Bug fix'].iloc[1])