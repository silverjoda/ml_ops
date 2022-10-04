import pandas as pd
import numpy as np

arr = np.array([[1, 2], [4, 5], [7, 8]])
print(arr.shape)
df2 = pd.DataFrame(arr,columns=['a', 'b', 'c'])