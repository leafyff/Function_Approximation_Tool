import sympy as sp
from fitting import FittedModel
import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline


def model_to_latex(model: FittedModel) -> str:
    x = sp.Symbol('x')

    if model.latex_repr == "polynomial":
        coeffs = model.params["coeffs"]
        degree = model.params["degree"]

        expr = sum(sp.Rational(float(coeffs[i])).limit_denominator(10000) * x ** (degree - i)
                   for i in range(len(coeffs)))
        expr = sp.simplify(expr)

        latex = sp.latex(expr)
        return f"$f(x) = {latex}$"

    elif model.latex_repr == "sinusoidal":
        A = model.params["A"]
        f = model.params["f"]
        phase = model.params["phase"]
        B = model.params["B"]

        A_sym = sp.Rational(float(A)).limit_denominator(10000)
        f_sym = sp.Rational(float(f)).limit_denominator(10000)
        phase_sym = sp.Rational(float(phase)).limit_denominator(10000)
        B_sym = sp.Rational(float(B)).limit_denominator(10000)

        expr = A_sym * sp.sin(2 * sp.pi * f_sym * x + phase_sym) + B_sym
        expr = sp.simplify(expr)

        latex = sp.latex(expr)
        return f"$f(x) = {latex}$"

    elif model.latex_repr == "exponential":
        A = model.params["A"]
        B = model.params["B"]
        C = model.params["C"]

        A_sym = sp.Rational(float(A)).limit_denominator(10000)
        B_sym = sp.Rational(float(B)).limit_denominator(10000)
        C_sym = sp.Rational(float(C)).limit_denominator(10000)

        expr = A_sym * sp.exp(B_sym * x) + C_sym
        expr = sp.simplify(expr)

        latex = sp.latex(expr)
        return f"$f(x) = {latex}$"

    elif model.latex_repr == "logarithmic":
        A = model.params["A"]
        B = model.params["B"]

        A_sym = sp.Rational(float(A)).limit_denominator(10000)
        B_sym = sp.Rational(float(B)).limit_denominator(10000)

        expr = A_sym * sp.ln(x) + B_sym
        expr = sp.simplify(expr)

        latex = sp.latex(expr)
        return f"$f(x) = {latex}$"

    elif model.latex_repr == "piecewise":
        segments = model.params["segments"]

        cases = []
        for x_min, x_max, seg_model in segments:
            seg_latex = model_to_latex(seg_model)
            seg_latex = seg_latex.replace("$f(x) = ", "").replace("$", "")

            x_min_str = f"{x_min:.2f}"
            x_max_str = f"{x_max:.2f}"
            cases.append(f"{seg_latex} & x \\in [{x_min_str}, {x_max_str}]")

        cases_str = " \\\\\n".join(cases)
        return f"$$f(x) = \\begin{{cases}}\n{cases_str}\n\\end{{cases}}$$"

    elif model.latex_repr == "spline":
        spline = model.params.get("spline")

        if spline is not None:
            if isinstance(spline, CubicSpline):
                return _spline_to_latex_from_cubic(spline)
            elif isinstance(spline, UnivariateSpline):
                return _spline_to_latex_from_univariate(spline)

        n_segments = model.params.get("n_segments", "unknown")
        return f"$f(x) = \\text{{Cubic spline with {n_segments} segments}}$"

    return "$f(x) = \\text{unknown}$"


def _spline_to_latex_from_cubic(spline: CubicSpline) -> str:
    x_breaks = spline.x
    coeffs = spline.c

    n_segments = len(x_breaks) - 1

    if n_segments > 8:
        x_sample = np.linspace(x_breaks[0], x_breaks[-1], 200)
        y_sample = spline(x_sample)

        poly_coeffs = np.polyfit(x_sample, y_sample, min(10, len(x_sample) - 1))

        x_sym = sp.Symbol('x')
        expr = sum(sp.Rational(float(poly_coeffs[i])).limit_denominator(10000) * x_sym ** (len(poly_coeffs) - 1 - i)
                   for i in range(len(poly_coeffs)))
        expr = sp.simplify(expr)

        latex = sp.latex(expr)
        return f"$f(x) \\approx {latex}$ (simplified from {n_segments} segments)"

    cases = []
    x_sym = sp.Symbol('x')

    for i in range(n_segments):
        x_min = x_breaks[i]
        x_max = x_breaks[i + 1]

        c3, c2, c1, c0 = coeffs[0, i], coeffs[1, i], coeffs[2, i], coeffs[3, i]

        expr = (sp.Rational(float(c3)).limit_denominator(10000) * (x_sym - x_min) ** 3 +
                sp.Rational(float(c2)).limit_denominator(10000) * (x_sym - x_min) ** 2 +
                sp.Rational(float(c1)).limit_denominator(10000) * (x_sym - x_min) +
                sp.Rational(float(c0)).limit_denominator(10000))

        expr = sp.expand(expr)
        expr = sp.simplify(expr)

        latex_expr = sp.latex(expr)
        x_min_str = f"{x_min:.3f}"
        x_max_str = f"{x_max:.3f}"

        cases.append(f"{latex_expr} & x \\in [{x_min_str}, {x_max_str}]")

    cases_str = " \\\\\n".join(cases)
    return f"$$f(x) = \\begin{{cases}}\n{cases_str}\n\\end{{cases}}$$"


def _spline_to_latex_from_univariate(spline: UnivariateSpline) -> str:
    knots = spline.get_knots()

    if len(knots) > 10:
        x_sample = np.linspace(knots[0], knots[-1], 200)
        y_sample = spline(x_sample)

        poly_coeffs = np.polyfit(x_sample, y_sample, min(10, len(x_sample) - 1))

        x_sym = sp.Symbol('x')
        expr = sum(sp.Rational(float(poly_coeffs[i])).limit_denominator(10000) * x_sym ** (len(poly_coeffs) - 1 - i)
                   for i in range(len(poly_coeffs)))
        expr = sp.simplify(expr)

        latex = sp.latex(expr)
        return f"$f(x) \\approx {latex}$ (simplified from spline)"

    try:
        tck = spline._eval_args
        t, c, k = tck

        from scipy.interpolate import PPoly
        pp = PPoly.from_spline((t, c, k))

        x_breaks = pp.x
        coeffs = pp.c
        n_segments = len(x_breaks) - 1

        if n_segments > 8:
            x_sample = np.linspace(x_breaks[0], x_breaks[-1], 200)
            y_sample = pp(x_sample)

            poly_coeffs = np.polyfit(x_sample, y_sample, min(10, len(x_sample) - 1))

            x_sym = sp.Symbol('x')
            expr = sum(sp.Rational(float(poly_coeffs[i])).limit_denominator(10000) * x_sym ** (len(poly_coeffs) - 1 - i)
                       for i in range(len(poly_coeffs)))
            expr = sp.simplify(expr)

            latex = sp.latex(expr)
            return f"$f(x) \\approx {latex}$ (simplified from spline)"

        cases = []
        x_sym = sp.Symbol('x')

        for i in range(n_segments):
            x_min = x_breaks[i]
            x_max = x_breaks[i + 1]

            poly_coeffs_seg = coeffs[:, i]

            expr = sum(sp.Rational(float(poly_coeffs_seg[j])).limit_denominator(10000) * (x_sym - x_min) ** (
                        len(poly_coeffs_seg) - 1 - j)
                       for j in range(len(poly_coeffs_seg)))

            expr = sp.expand(expr)
            expr = sp.simplify(expr)

            latex_expr = sp.latex(expr)
            x_min_str = f"{x_min:.3f}"
            x_max_str = f"{x_max:.3f}"

            cases.append(f"{latex_expr} & x \\in [{x_min_str}, {x_max_str}]")

        cases_str = " \\\\\n".join(cases)
        return f"$$f(x) = \\begin{{cases}}\n{cases_str}\n\\end{{cases}}$$"
    except:
        return f"$f(x) = \\text{{Cubic spline with {len(knots)} knots}}$"