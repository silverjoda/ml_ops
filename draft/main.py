import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data, index = ["day1", "day2", "day3"])

print(df)

df2 = df[0:2]

print(df2)

df3 = df2.drop('day1')

print(df)

print(df2)

print(df3)