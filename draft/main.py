import pandas as pd
import numpy as np

arr = np.array([[1, 2], [4, 5], [7, 8], [5, 2], [1, 1]])
df = pd.DataFrame(arr,columns=['a', 'b'])

data_ser = df['a']
print(data_ser)
data_ser = pd.Series(data=data_ser.values, index=data_ser.index + 10)
print(data_ser)
