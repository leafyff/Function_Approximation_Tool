
https://github.com/leafyff/Function_Approximation_Tool


### First plan: (09.01.2026)

- Window display (Qt)
- Tools for comfortable drawing must be available
- Smoothing of drawings
- Reading and evaluation of all function options
- Generation of LateX code based on the read data (TikZ)


### v0.1 (21.01.2026)

The following steps have been implemented:
- Window display (Qt)
- Smoothing of the drawn image
- Function reading:
	- Polynomial approximation,
	- Cubic splines
	- Approximation with basic functions: exponential, logarithmic, sinusoidal
	- Efficiency evaluation: RMSE
- LateX code generation based on the read data

### v0.2 (12.02.2026)

New methods for analyzing functions are now available:
- Polynomial Approximation (Interpolation) (Chebyshev Basis)
- Cubic Splines 
- AAA Algorithm (Adaptive Antoulas-Anderson) (Nakatsukasa–Sète–Trefethen)
- Discrete Minimax via Linear Programming
- Simple functions: exponential, logarithmic, hyperbolic, sinusoidal, tangentoidal
New metrics for analyzing approximation quality:
- RMSE - Root mean square error
- L-infinity / Minimax Error - Worst-case error
- BIC (Bayesian Information Criterion)
- Multi-objective Score - Combined criterion for the three above to give a single rating

### v0.3  (19.02.2026)

- List of created models:
	1. Cubic Spline
	2. Interpolation polynomial (Chebyshev Basis)
	3. L-inf minimax polynomial
	4. Polynomial Least Squares Approximation  (Chebyshev Basis)
	5. Non-Uniform Fast Fourier Transform (NUFFT)
	6. AAA Algorithm
	7. exponential curve
	8. logarithmic curve
	9. rational curve
	10. sinusoidal curve
	11. tangential curve
	12. arctan (S-curve)
- Improved scoring
- Improved fitting


### Можливо буде реалізовано (long-term plans):
- Генерація напряму TikZ коду 
- Скорочення LateX коду
- Інструменти для гарного малювання
- Додати можливість задавати кількість вузлів інтерполяції
- Графічний інтерфейс напряму в GitHub (Веб Дизайн)
- Аналіз параметричних кривих
- ШІ аналіз?