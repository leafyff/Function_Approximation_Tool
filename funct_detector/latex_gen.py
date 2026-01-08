import sympy as sp
from fitting import FittedModel
import numpy as np


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

        if spline is not None and hasattr(spline, 'get_knots'):
            knots = spline.get_knots()
            n_knots = len(knots)

            if n_knots > 10:
                x_sample = np.linspace(knots[0], knots[-1], 100)
                y_sample = spline(x_sample)

                coeffs = np.polyfit(x_sample, y_sample, min(8, len(x_sample) - 1))

                expr = sum(sp.Rational(float(coeffs[i])).limit_denominator(10000) * x ** (len(coeffs) - 1 - i)
                           for i in range(len(coeffs)))
                expr = sp.simplify(expr)

                latex = sp.latex(expr)
                return f"$f(x) \\approx {latex}$ (spline approximation, {n_knots} knots)"
            else:
                return f"$f(x) = \\text{{Cubic spline with {n_knots} knots}}$"
        else:
            n_segments = model.params.get("n_segments", "many")
            return f"$f(x) = \\text{{Cubic spline with {n_segments} segments}}$"

    return "$f(x) = \\text{unknown}$"

