# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date: 18-03-2025
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/SEC/Downloads/dc.csv", parse_dates=['Date'], index_col='Date', dayfirst=True)

data.head()

resampled_data = data['open_USD'].resample('YE').sum().to_frame()
resampled_data.head()

resampled_data.index = resampled_data.index.year

resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'Date': 'Year'}, inplace=True)

resampled_data.head()

years = resampled_data['Year'].tolist()
open_usd = resampled_data['open_USD'].tolist() 

```
A - LINEAR TREND ESTIMATION
```
X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, open_usd)] 

n = len(years)
b = (n * sum(xy) - sum(open_usd) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(open_usd) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

```

B- POLYNOMIAL TREND ESTIMATION
```
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, open_usd)] 

coeff = [[len(X), sum(X), sum(x2)],
[sum(X), sum(x2), sum(x3)],
[sum(x2), sum(x3), sum(x4)]]

Y = [sum(open_usd), sum(xy), sum(x2y)] 
A = np.array(coeff)
B = np.array(Y)

solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]
```
C- VISUALISING RESULTS
```
print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"\nPolynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend

resampled_data.set_index('Year',inplace=True)

resampled_data['open_USD'].plot(kind='line', color='blue', marker='o') 
resampled_data['Linear Trend'].plot(kind='line', color='black', linestyle='--')

resampled_data['open_USD'].plot(kind='line',color='blue',marker='o')
resampled_data['Polynomial Trend'].plot(kind='line',color='black',marker='o')
```
### OUTPUT
A - LINEAR TREND ESTIMATION

![image](https://github.com/user-attachments/assets/619d7812-c871-4473-86b3-c6366cc098e3)


B- POLYNOMIAL TREND ESTIMATION

![image](https://github.com/user-attachments/assets/e50d96e9-49b3-4424-981e-66671f9f4300)


C- TREND EQUATIONS

![image](https://github.com/user-attachments/assets/1b2e1b5d-a713-474a-a400-bf640e10ded5)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
