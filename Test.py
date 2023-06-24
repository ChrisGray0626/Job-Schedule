# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2023/6/24
"""

import pandas as pd

def dataframe():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])
    print(df)
    a = df.loc[0]
    a['a'] = 10
    print(df)

if __name__ == '__main__':
    dataframe()
    pass
