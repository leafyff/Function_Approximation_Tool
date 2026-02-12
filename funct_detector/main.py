"""
Function Drawer and LaTeX Generator.

Draw a curve in a coordinate system, approximate it as y = f(x), and show a LaTeX
representation of the best fitting model.
"""

from __future__ import annotations

import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pyqtgraph as pg
import sympy as sp
from numpy.typing import NDArray
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QCursor, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.interpolate import CubicSpline, UnivariateSpline, interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

FloatArray = NDArray[np.floating[Any]]
EvaluationFunction = Callable[[FloatArray], FloatArray]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PlotSettings:
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    grid_spacing: float = 1.0
    accuracy: float = 0.0001

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be less than x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be less than y_max ({self.y_max})")
        if self.grid_spacing <= 0:
            raise ValueError(f"grid_spacing ({self.grid_spacing}) must be positive")
        if not (0.0001 <= self.accuracy <= 1.0):
            raise ValueError(f"accuracy ({self.accuracy}) must be between 0.0001 and 1.0")

    @property
    def domain_width(self) -> float:
        return self.x_max - self.x_min

    @property
    def domain_height(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True, slots=True)
class FittedModel:
    name: str
    evaluate: EvaluationFunction
    latex_kind: str
    rmse: float
    aic: float
    complexity: float
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rmse < 0:
            raise ValueError(f"RMSE cannot be negative: {self.rmse}")
        if self.complexity < 0:
            raise ValueError(f"Complexity cannot be negative: {self.complexity}")
        if not callable(self.evaluate):
            raise ValueError("evaluate must be callable")


# ---------------------------------------------------------------------------
# Model fitters
# ---------------------------------------------------------------------------

class ModelFitter(ABC):
    @abstractmethod
    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        raise NotImplementedError

    @staticmethod
    def _rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _aic(n: int, mse: float, k: int) -> float:
        if mse <= 0:
            return float("inf")
        return float(n * np.log(mse) + 2 * k)

    @staticmethod
    def _linear_fit_r2(x: FloatArray, y: FloatArray) -> tuple[FloatArray, float]:
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        y_var = float(np.var(y))
        if y_var <= 0:
            return coeffs, 0.0
        r2 = 1.0 - float(np.var(y - y_pred)) / y_var
        return coeffs, r2


class PolynomialFitter(ModelFitter):
    def __init__(self, max_degree: int = 15) -> None:
        self._max_degree = max_degree

    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        best_degree: Optional[int] = None
        best_aic: float = float("inf")
        best_coeffs: Optional[FloatArray] = None

        target_rmse = max(0.0001, float(accuracy))
        max_degree = min(self._max_degree, max(1, len(x) - 1))

        for degree in range(1, max_degree + 1):
            try:
                coeffs = np.polyfit(x, y, degree)
                y_pred = np.asarray(np.polyval(coeffs, x), dtype=np.float64)
                rmse = self._rmse(y, y_pred)

                mse = rmse * rmse
                aic = self._aic(len(x), mse, degree + 1)

                penalty = 0.5 * degree if degree > 8 else 0.0
                score = aic + penalty

                if score < best_aic:
                    best_aic = score
                    best_degree = degree
                    best_coeffs = coeffs

                if rmse < target_rmse and degree > 1:
                    break
            except (np.linalg.LinAlgError, ValueError):
                continue

        if best_coeffs is None or best_degree is None:
            return None

        # Capture as plain Python objects to avoid the
        # "ndarray | iterable | int | float" subscript warnings
        # that arise when numpy arrays are closed over directly.
        captured_coeffs: list[float] = [float(c) for c in best_coeffs]
        y_pred_best = np.asarray(np.polyval(captured_coeffs, x), dtype=np.float64)
        rmse_best = self._rmse(y, y_pred_best)

        def evaluate(x_eval: FloatArray) -> FloatArray:
            return np.asarray(np.polyval(captured_coeffs, x_eval), dtype=np.float64)

        return FittedModel(
            name=f"Polynomial (degree {best_degree})",
            evaluate=evaluate,
            latex_kind="polynomial",
            rmse=rmse_best,
            aic=best_aic,
            complexity=float(best_degree) * 0.5,
            params={"coeffs": captured_coeffs, "degree": best_degree},
        )


class SinusoidalFitter(ModelFitter):
    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        try:
            dx = np.diff(x)
            if len(dx) == 0:
                return None
            step = float(np.mean(dx))
            if not np.isfinite(step) or step == 0.0:
                return None

            y_centered = y - np.mean(y)
            y_fft = np.fft.fft(y_centered)
            freqs = np.fft.fftfreq(len(x), step)

            half = max(1, len(freqs) // 2)
            positive_freqs = freqs[:half]
            positive_fft = np.abs(y_fft[:half])

            if len(positive_fft) < 2:
                return None

            dominant_idx = int(np.argmax(positive_fft[1:]) + 1)
            f0 = float(np.abs(positive_freqs[dominant_idx]))
            if f0 == 0.0:
                return None

            total_power = float(np.sum(positive_fft))
            if total_power <= 0:
                return None
            power_ratio = float(positive_fft[dominant_idx]) / total_power
            if power_ratio < 0.4:
                return None

            a_init = float(2 * positive_fft[dominant_idx] / len(x))
            b_init = float(np.mean(y))
            phase_init = 0.0

            def sin_func(
                x_vals: FloatArray,
                amplitude: float,
                freq: float,
                phase: float,
                offset: float,
            ) -> FloatArray:
                return np.asarray(
                    amplitude * np.sin(2 * np.pi * freq * x_vals + phase) + offset,
                    dtype=np.float64,
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_result = curve_fit(
                    sin_func,
                    x,
                    y,
                    p0=[a_init, f0, phase_init, b_init],
                    maxfev=5000,
                )
            popt: FloatArray = np.asarray(fit_result[0], dtype=np.float64)

            # Unpack to plain Python floats to avoid ndarray subscript warnings
            amp = float(popt[0])
            freq = float(popt[1])
            phase = float(popt[2])
            off = float(popt[3])

            y_pred = sin_func(x, amp, freq, phase, off)
            rmse = self._rmse(y, y_pred)

            target_rmse = max(0.0001, float(accuracy))
            y_std = float(np.std(y))
            if y_std > 0 and (rmse / y_std > 0.5 or rmse > target_rmse * 10):
                return None

            aic = self._aic(len(x), rmse * rmse, 4)

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return sin_func(x_eval, amp, freq, phase, off)

            return FittedModel(
                name="Sinusoidal",
                evaluate=evaluate,
                latex_kind="sinusoidal",
                rmse=rmse,
                aic=aic,
                complexity=3.0,
                params={"A": amp, "f": freq, "phase": phase, "B": off},
            )
        except (
            RuntimeError,
            ValueError,
            TypeError,
            IndexError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ):
            return None


class ExponentialFitter(ModelFitter):
    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        try:
            if np.all(y > 0):
                y_log = np.log(y)
                sign = 1.0
            elif np.all(y < 0):
                y_log = np.log(-y)
                sign = -1.0
            else:
                return None

            coeffs, r2 = self._linear_fit_r2(x, y_log)
            if r2 < 0.85:
                return None

            def exp_func(
                x_vals: FloatArray, amplitude: float, rate: float, offset: float
            ) -> FloatArray:
                return np.asarray(
                    amplitude * np.exp(rate * x_vals) + offset, dtype=np.float64
                )

            a_init = float(sign * np.exp(float(coeffs[1])))
            b_init = float(coeffs[0])
            c_init = 0.0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_result = curve_fit(
                    exp_func,
                    x,
                    y,
                    p0=[a_init, b_init, c_init],
                    maxfev=5000,
                )
            popt: FloatArray = np.asarray(fit_result[0], dtype=np.float64)

            amp = float(popt[0])
            rate = float(popt[1])
            off = float(popt[2])

            y_pred = exp_func(x, amp, rate, off)
            rmse = self._rmse(y, y_pred)
            target_rmse = max(0.0001, float(accuracy))
            if rmse > target_rmse * 10:
                return None
            aic = self._aic(len(x), rmse * rmse, 3)

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return exp_func(x_eval, amp, rate, off)

            return FittedModel(
                name="Exponential",
                evaluate=evaluate,
                latex_kind="exponential",
                rmse=rmse,
                aic=aic,
                complexity=3.0,
                params={"A": amp, "B": rate, "C": off},
            )
        except (
            RuntimeError,
            ValueError,
            TypeError,
            OverflowError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ):
            return None


class LogarithmicFitter(ModelFitter):
    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        try:
            if not np.all(x > 0):
                return None

            x_log = np.log(x)
            coeffs, r2 = self._linear_fit_r2(x_log, y)
            if r2 < 0.85:
                return None

            def log_func(x_vals: FloatArray, amplitude: float, offset: float) -> FloatArray:
                return np.asarray(
                    amplitude * np.log(x_vals) + offset, dtype=np.float64
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_result = curve_fit(
                    log_func,
                    x,
                    y,
                    p0=[float(coeffs[0]), float(coeffs[1])],
                )
            popt: FloatArray = np.asarray(fit_result[0], dtype=np.float64)

            amp = float(popt[0])
            off = float(popt[1])

            y_pred = log_func(x, amp, off)
            rmse = self._rmse(y, y_pred)
            target_rmse = max(0.0001, float(accuracy))
            if rmse > target_rmse * 10:
                return None
            aic = self._aic(len(x), rmse * rmse, 2)

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return log_func(x_eval, amp, off)

            return FittedModel(
                name="Logarithmic",
                evaluate=evaluate,
                latex_kind="logarithmic",
                rmse=rmse,
                aic=aic,
                complexity=3.0,
                params={"A": amp, "B": off},
            )
        except (
            RuntimeError,
            ValueError,
            TypeError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ):
            return None


class SplineFitter(ModelFitter):
    def fit(self, x: FloatArray, y: FloatArray, accuracy: float) -> Optional[FittedModel]:
        sorted_indices = np.argsort(x)
        x_sorted: FloatArray = x[sorted_indices]
        y_sorted: FloatArray = y[sorted_indices]

        noise_estimate = (
            float(np.std(np.diff(y_sorted)) / np.sqrt(2)) if len(y_sorted) > 1 else 0.1
        )
        s_base = len(x_sorted) * noise_estimate * noise_estimate
        s = s_base * (float(accuracy) / 0.01)

        try:
            spline = UnivariateSpline(x_sorted, y_sorted, s=s, k=3)
            y_pred = np.asarray(spline(x_sorted), dtype=np.float64)
            rmse = self._rmse(y_sorted, y_pred)

            knots = np.asarray(spline.get_knots(), dtype=np.float64)
            k_count = int(len(knots))
            aic = self._aic(len(x_sorted), rmse * rmse, k_count + 4)

            def evaluate(x_eval: FloatArray) -> FloatArray:
                return np.asarray(spline(x_eval), dtype=np.float64)

            return FittedModel(
                name=f"Cubic spline ({k_count} knots)",
                evaluate=evaluate,
                latex_kind="spline",
                rmse=rmse,
                aic=aic,
                complexity=float(k_count) * 0.3,
                params={"spline": spline, "knots": knots, "segments": k_count},
            )
        except (ValueError, TypeError):
            spline_cs = CubicSpline(x_sorted, y_sorted)
            y_pred = np.asarray(spline_cs(x_sorted), dtype=np.float64)
            rmse = self._rmse(y_sorted, y_pred)

            segments = max(1, len(x_sorted) - 1)
            aic = self._aic(len(x_sorted), rmse * rmse, segments * 4)

            def evaluate(x_eval: FloatArray) -> FloatArray:  # type: ignore[misc]
                return np.asarray(spline_cs(x_eval), dtype=np.float64)

            return FittedModel(
                name=f"Cubic spline ({segments} segments)",
                evaluate=evaluate,
                latex_kind="spline",
                rmse=rmse,
                aic=aic,
                complexity=float(segments) * 0.3,
                params={"spline": spline_cs, "segments": segments},
            )


# ---------------------------------------------------------------------------
# Model selection service
# ---------------------------------------------------------------------------

class ModelSelectionService:
    def __init__(self) -> None:
        self._fitters: tuple[ModelFitter, ...] = (
            PolynomialFitter(),
            SinusoidalFitter(),
            ExponentialFitter(),
            LogarithmicFitter(),
            SplineFitter(),
        )

    def fit_all_models(self, x: FloatArray, y: FloatArray, accuracy: float) -> list[FittedModel]:
        models: list[FittedModel] = []
        for fitter in self._fitters:
            try:
                model = fitter.fit(x, y, accuracy)
                if model is not None:
                    models.append(model)
            except (RuntimeError, ValueError, TypeError, OverflowError, FloatingPointError):
                continue
        return models

    @staticmethod
    def select_best_model(models: list[FittedModel], y: FloatArray) -> FittedModel:
        if not models:
            raise ValueError("No models to select from")

        y_std = float(np.std(y))

        def score(m: FittedModel) -> float:
            normalized_rmse = m.rmse / y_std if y_std > 0 else m.rmse
            return float(normalized_rmse + 0.1 * m.aic + m.complexity)

        scores = np.asarray([score(m) for m in models], dtype=np.float64)
        best_idx = int(np.argmin(scores))

        if len(models) > 1:
            sorted_indices = np.argsort(scores)
            second_idx = int(sorted_indices[1])
            best_score = float(scores[best_idx])
            second_score = float(scores[second_idx])
            if best_score > 0 and abs(best_score - second_score) < 0.05 * best_score:
                if models[best_idx].complexity > models[second_idx].complexity:
                    return models[second_idx]

        return models[best_idx]


# ---------------------------------------------------------------------------
# LaTeX generator
# ---------------------------------------------------------------------------

class LaTeXGenerator:
    def generate(self, model: FittedModel) -> str:
        try:
            if model.latex_kind == "polynomial":
                return self._polynomial(model)
            if model.latex_kind == "sinusoidal":
                return self._sinusoidal(model)
            if model.latex_kind == "exponential":
                return self._exponential(model)
            if model.latex_kind == "logarithmic":
                return self._logarithmic(model)
            if model.latex_kind == "spline":
                return self._spline(model)
            return r"$f(x) = \text{unknown}$"
        except (ValueError, TypeError, AttributeError):
            return r"$f(x) = \text{error}$"

    @staticmethod
    def _wrap(expr: sp.Basic) -> str:
        simplified = sp.simplify(expr)
        return f"$f(x) = {sp.latex(simplified)}$"

    def _polynomial(self, model: FittedModel) -> str:
        raw_coeffs = model.params["coeffs"]
        degree = int(model.params["degree"])
        # Accept either a list[float] (new) or ndarray (legacy)
        coeffs: list[float] = (
            [float(c) for c in raw_coeffs]
            if hasattr(raw_coeffs, "__iter__")
            else [float(raw_coeffs)]
        )

        x = sp.Symbol("x")
        terms: list[sp.Basic] = []
        for i, c in enumerate(coeffs):
            exp = degree - i
            coef = sp.Rational(c).limit_denominator(10000)
            terms.append(coef * x ** exp)
        return self._wrap(sp.Add(*terms))

    def _sinusoidal(self, model: FittedModel) -> str:
        a_val = float(model.params["A"])
        f_val = float(model.params["f"])
        phase = float(model.params["phase"])
        b_val = float(model.params["B"])

        x = sp.Symbol("x")
        a = sp.Rational(a_val).limit_denominator(10000)
        f = sp.Rational(f_val).limit_denominator(10000)
        ph = sp.Rational(phase).limit_denominator(10000)
        b = sp.Rational(b_val).limit_denominator(10000)

        expr = a * sp.sin(2 * sp.pi * f * x + ph) + b
        return self._wrap(expr)

    def _exponential(self, model: FittedModel) -> str:
        a_val = float(model.params["A"])
        b_val = float(model.params["B"])
        c_val = float(model.params["C"])

        x = sp.Symbol("x")
        a = sp.Rational(a_val).limit_denominator(10000)
        b = sp.Rational(b_val).limit_denominator(10000)
        c = sp.Rational(c_val).limit_denominator(10000)

        expr = a * sp.exp(b * x) + c
        return self._wrap(expr)

    def _logarithmic(self, model: FittedModel) -> str:
        a_val = float(model.params["A"])
        b_val = float(model.params["B"])

        x = sp.Symbol("x")
        a = sp.Rational(a_val).limit_denominator(10000)
        b = sp.Rational(b_val).limit_denominator(10000)

        expr = a * sp.ln(x) + b
        return self._wrap(expr)

    @staticmethod
    def _spline(model: FittedModel) -> str:
        segments = model.params.get("segments", "unknown")
        return rf"$f(x) = \text{{Cubic spline with {segments} segments}}$"


# ---------------------------------------------------------------------------
# Stroke preprocessor
# ---------------------------------------------------------------------------

class StrokePreprocessor:
    def __init__(self, domain_width: float, domain_height: float) -> None:
        if domain_width <= 0 or domain_height <= 0:
            raise ValueError("Domain dimensions must be positive")
        self._domain_width = float(domain_width)
        self._domain_height = float(domain_height)

    def preprocess(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        n_samples = max(100, int(self._domain_width * 50))
        x_res, y_res = self._resample_by_arc_length(x, y, n_samples)
        x_smooth, y_smooth = self._smooth_curve(x_res, y_res)
        x_clean, y_clean = self._remove_outliers(x_smooth, y_smooth)
        if len(x_clean) > 50:
            x_clean, y_clean = self._simplify_curve(x_clean, y_clean)
        return x_clean, y_clean

    def is_function(self, x: FloatArray, y: FloatArray) -> tuple[bool, str]:
        epsilon = self._domain_width / 1000.0
        sorted_indices = np.argsort(x)
        x_sorted: FloatArray = x[sorted_indices]
        y_sorted: FloatArray = y[sorted_indices]

        j = 0
        for i in range(len(x_sorted)):
            if j < i:
                j = i
            while j + 1 < len(x_sorted) and x_sorted[j + 1] - x_sorted[i] <= epsilon:
                j += 1
            if j > i:
                y_window = y_sorted[i:j + 1]
                if float(np.max(y_window) - np.min(y_window)) > epsilon:
                    return False, f"Multiple y-values at x ≈ {float(x_sorted[i]):.2f}"
        return True, ""

    @staticmethod
    def _resample_by_arc_length(
        x: FloatArray, y: FloatArray, n_samples: int
    ) -> tuple[FloatArray, FloatArray]:
        if len(x) < 2:
            return x, y

        dx = np.diff(x)
        dy = np.diff(y)
        s: FloatArray = np.concatenate(([0.0], np.cumsum(np.sqrt(dx * dx + dy * dy))))
        if float(s[-1]) == 0.0:
            return x, y

        s_uniform = np.linspace(0.0, float(s[-1]), int(n_samples))

        try:
            f_x = interp1d(s, x, kind="linear", fill_value="extrapolate")
            f_y = interp1d(s, y, kind="linear", fill_value="extrapolate")
            return (
                np.asarray(f_x(s_uniform), dtype=np.float64),
                np.asarray(f_y(s_uniform), dtype=np.float64),
            )
        except (ValueError, IndexError):
            return x, y

    @staticmethod
    def _smooth_curve(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        if len(x) < 15:
            return x, y

        window = len(x) if len(x) % 2 == 1 else len(x) - 1
        window = min(15, window)
        if window < 3:
            return x, y

        try:
            poly_order = min(3, window - 1)
            y_smooth = savgol_filter(y, window, poly_order)
            return x, np.asarray(y_smooth, dtype=np.float64)
        except (ValueError, TypeError):
            return x, y

    @staticmethod
    def _remove_outliers(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        if len(x) < 5:
            return x, y

        curvatures = np.zeros(len(x), dtype=np.float64)
        radius = max(2, int(len(x) * 0.02))

        for i in range(len(x)):
            i_prev = max(0, i - radius)
            i_next = min(len(x) - 1, i + radius)
            if i_next - i_prev < 2:
                continue

            pts = np.array(
                [[x[i_prev], y[i_prev]], [x[i], y[i]], [x[i_next], y[i_next]]],
                dtype=np.float64,
            )
            a = float(np.linalg.norm(pts[1] - pts[0]))
            b = float(np.linalg.norm(pts[2] - pts[1]))
            c = float(np.linalg.norm(pts[2] - pts[0]))
            denominator = a * b * c
            if denominator <= 1e-10:
                continue

            area = 0.5 * abs(float(np.cross(pts[1] - pts[0], pts[2] - pts[0])))
            curvatures[i] = 4.0 * area / denominator

        threshold = float(np.median(curvatures) + 3.0 * np.std(curvatures))
        mask = curvatures < threshold
        return x[mask], y[mask]

    def _simplify_curve(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        points = np.column_stack((x, y))
        tolerance = 0.01 * self._domain_height
        simplified = self._simplify_line(points, float(tolerance))
        return simplified[:, 0], simplified[:, 1]

    def _simplify_line(self, points: FloatArray, tolerance: float) -> FloatArray:
        if len(points) < 3:
            return points

        max_distance = 0.0
        split_index = 0
        for i in range(1, len(points) - 1):
            dist = self._perpendicular_distance(points[i], points[0], points[-1])
            if dist > max_distance:
                split_index = i
                max_distance = dist

        if max_distance > tolerance:
            first = self._simplify_line(points[: split_index + 1], tolerance)
            second = self._simplify_line(points[split_index:], tolerance)
            return np.vstack((first[:-1], second))

        return np.array([points[0], points[-1]], dtype=np.float64)

    @staticmethod
    def _perpendicular_distance(
        point: FloatArray, line_start: FloatArray, line_end: FloatArray
    ) -> float:
        if np.allclose(line_start, line_end):
            return float(np.linalg.norm(point - line_start))
        numerator = abs(float(np.cross(line_end - line_start, line_start - point)))
        denominator = float(np.linalg.norm(line_end - line_start))
        if denominator == 0.0:
            return 0.0
        return numerator / denominator


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    def __init__(self, current_settings: PlotSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot Settings")
        self._settings = current_settings
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QGridLayout()

        self._x_min_edit = QLineEdit(str(self._settings.x_min))
        self._x_max_edit = QLineEdit(str(self._settings.x_max))
        self._y_min_edit = QLineEdit(str(self._settings.y_min))
        self._y_max_edit = QLineEdit(str(self._settings.y_max))
        self._grid_edit = QLineEdit(str(self._settings.grid_spacing))

        self._accuracy_spinbox = QDoubleSpinBox()
        self._accuracy_spinbox.setRange(0.0001, 1.0)
        self._accuracy_spinbox.setSingleStep(0.0001)
        self._accuracy_spinbox.setDecimals(4)
        self._accuracy_spinbox.setValue(self._settings.accuracy)

        layout.addWidget(QLabel("X Min:"), 0, 0)
        layout.addWidget(self._x_min_edit, 0, 1)
        layout.addWidget(QLabel("X Max:"), 1, 0)
        layout.addWidget(self._x_max_edit, 1, 1)
        layout.addWidget(QLabel("Y Min:"), 2, 0)
        layout.addWidget(self._y_min_edit, 2, 1)
        layout.addWidget(QLabel("Y Max:"), 3, 0)
        layout.addWidget(self._y_max_edit, 3, 1)
        layout.addWidget(QLabel("Grid Spacing:"), 4, 0)
        layout.addWidget(self._grid_edit, 4, 1)
        layout.addWidget(QLabel("Approximation Accuracy:"), 5, 0)
        layout.addWidget(self._accuracy_spinbox, 5, 1)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout, 6, 0, 1, 2)
        self.setLayout(layout)

    def get_settings(self) -> Optional[PlotSettings]:
        try:
            return PlotSettings(
                x_min=float(self._x_min_edit.text()),
                x_max=float(self._x_max_edit.text()),
                y_min=float(self._y_min_edit.text()),
                y_max=float(self._y_max_edit.text()),
                grid_spacing=float(self._grid_edit.text()),
                accuracy=float(self._accuracy_spinbox.value()),
            )
        except (ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class DrawingApp(QMainWindow):
    _COLORS: tuple[tuple[int, int, int], ...] = (
        (255, 0, 0),
        (0, 200, 0),
        (255, 150, 0),
        (200, 0, 200),
        (0, 200, 200),
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Function Drawer and LaTeX Generator")
        self.setGeometry(100, 100, 1400, 800)

        self._model_service = ModelSelectionService()
        self._latex = LaTeXGenerator()

        self._settings = PlotSettings()
        self._drawing = False
        self._strokes: list[list[tuple[float, float]]] = []
        self._current_stroke: Optional[list[tuple[float, float]]] = None
        self._panning = False
        self._pan_start_pos: Optional[QPointF] = None
        self._pan_start_range: Optional[tuple[tuple[float, float], tuple[float, float]]] = None

        self._fitted_curves: list[Any] = []
        self._drawn_curve: Optional[Any] = None
        self._shown_models: list[FittedModel] = []
        self._option_checkboxes: list[QCheckBox] = []
        self._option_widgets: list[QWidget] = []

        self._build_ui()
        self._configure_plot()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left_layout = QVBoxLayout()
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.addLegend(offset=(10, 10))
        left_layout.addWidget(self._plot_widget)

        button_layout = QHBoxLayout()
        self._clear_button = QPushButton("Clear")
        self._fit_button = QPushButton("Fit Curve")
        self._export_button = QPushButton("Copy LaTeX")
        self._settings_button = QPushButton("Settings")

        self._clear_button.clicked.connect(self.clear_drawing)
        self._fit_button.clicked.connect(self.fit_curve)
        self._export_button.clicked.connect(self.copy_latex)
        self._settings_button.clicked.connect(self.show_settings)

        button_layout.addWidget(self._clear_button)
        button_layout.addWidget(self._fit_button)
        button_layout.addWidget(self._export_button)
        button_layout.addWidget(self._settings_button)
        left_layout.addLayout(button_layout)

        main_layout.addLayout(left_layout, 3)

        right_layout = QVBoxLayout()

        options_group = QGroupBox("Display Options")
        self._options_layout = QVBoxLayout()
        options_group.setLayout(self._options_layout)

        self._clear_on_new_line = QCheckBox("Clear plot on a new line")
        self._clear_on_new_line.setChecked(True)
        self._options_layout.addWidget(self._clear_on_new_line)

        right_layout.addWidget(options_group)
        right_layout.addWidget(QLabel("LaTeX Output (Top Candidates):"))

        self._latex_output = QTextEdit()
        self._latex_output.setReadOnly(True)
        right_layout.addWidget(self._latex_output)

        main_layout.addLayout(right_layout, 1)

        vb = self._plot_widget.plotItem.vb
        vb.setMenuEnabled(False)
        vb.setMouseEnabled(x=True, y=True)
        self._plot_widget.viewport().installEventFilter(self)

    def _configure_plot(self) -> None:
        self._plot_widget.setLabel("left", "y")
        self._plot_widget.setLabel("bottom", "x")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)

        vb = self._plot_widget.plotItem.vb
        vb.disableAutoRange()
        self._plot_widget.setXRange(self._settings.x_min, self._settings.x_max)
        self._plot_widget.setYRange(self._settings.y_min, self._settings.y_max)

    # ------------------------------------------------------------------
    # Event filter
    # ------------------------------------------------------------------

    def eventFilter(self, obj: Any, event: QEvent) -> bool:  # noqa: N802
        if obj is not self._plot_widget.viewport():
            return super().eventFilter(obj, event)
        if not isinstance(event, QMouseEvent):
            return super().eventFilter(obj, event)

        vb = self._plot_widget.plotItem.vb

        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                if self._clear_on_new_line.isChecked():
                    self.clear_drawing()
                # Use position() (QPointF) instead of the deprecated pos() (QPoint)
                view_pos = vb.mapSceneToView(event.position())
                self._drawing = True
                self._current_stroke = [(float(view_pos.x()), float(view_pos.y()))]
                self._strokes.append(self._current_stroke)
                return True

            if event.button() == Qt.MouseButton.RightButton:
                self._panning = True
                self._pan_start_pos = vb.mapSceneToView(event.position())
                x_range: list[float] = vb.viewRange()[0]
                y_range: list[float] = vb.viewRange()[1]
                self._pan_start_range = (
                    (float(x_range[0]), float(x_range[1])),
                    (float(y_range[0]), float(y_range[1])),
                )
                self._plot_widget.viewport().setCursor(
                    QCursor(Qt.CursorShape.ClosedHandCursor)
                )
                return True

        if event.type() == QEvent.Type.MouseMove:
            if self._drawing and self._current_stroke is not None:
                view_pos = vb.mapSceneToView(event.position())
                self._current_stroke.append(
                    (float(view_pos.x()), float(view_pos.y()))
                )
                self.update_drawing()
                return True

            if (
                self._panning
                and self._pan_start_pos is not None
                and self._pan_start_range is not None
            ):
                current_pos = vb.mapSceneToView(event.position())
                dx = float(self._pan_start_pos.x() - current_pos.x())
                dy = float(self._pan_start_pos.y() - current_pos.y())

                x_range_start, y_range_start = self._pan_start_range
                vb.setRange(
                    xRange=(x_range_start[0] + dx, x_range_start[1] + dx),
                    yRange=(y_range_start[0] + dy, y_range_start[1] + dy),
                    padding=0,
                )
                return True

        if event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                self._drawing = False
                self._current_stroke = None
                return True

            if event.button() == Qt.MouseButton.RightButton:
                self._panning = False
                self._pan_start_pos = None
                self._pan_start_range = None
                self._plot_widget.viewport().unsetCursor()
                return True

        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def clear_drawing(self) -> None:
        self._strokes.clear()
        self._current_stroke = None
        self._drawing = False
        self._panning = False
        self._pan_start_pos = None
        self._pan_start_range = None

        self._plot_widget.viewport().unsetCursor()

        if self._drawn_curve is not None:
            self._plot_widget.removeItem(self._drawn_curve)
            self._drawn_curve = None

        for curve in self._fitted_curves:
            self._plot_widget.removeItem(curve)
        self._fitted_curves.clear()

        for widget in self._option_widgets:
            widget.setParent(None)  # type: ignore[call-overload]
            widget.deleteLater()
        self._option_widgets.clear()
        self._option_checkboxes.clear()

        self._latex_output.clear()
        self._shown_models.clear()

    def update_drawing(self) -> None:
        xs: list[float] = []
        ys: list[float] = []

        for stroke in self._strokes:
            if not stroke:
                continue
            if xs:
                xs.append(float("nan"))
                ys.append(float("nan"))
            sx, sy = zip(*stroke)
            xs.extend(sx)
            ys.extend(sy)

        if len(xs) < 2:
            if self._drawn_curve is not None:
                self._plot_widget.removeItem(self._drawn_curve)
                self._drawn_curve = None
            return

        x_arr = np.asarray(xs, dtype=np.float64)
        y_arr = np.asarray(ys, dtype=np.float64)

        if self._drawn_curve is None:
            self._drawn_curve = self._plot_widget.plot(
                x_arr,
                y_arr,
                pen=pg.mkPen((100, 100, 255), width=2),
                name="Drawn curve",
            )
        else:
            self._drawn_curve.setData(x_arr, y_arr)

    # ------------------------------------------------------------------
    # Curve fitting
    # ------------------------------------------------------------------

    def fit_curve(self) -> None:
        points: list[tuple[float, float]] = []
        for stroke in self._strokes:
            points.extend(stroke)

        if len(points) < 10:
            QMessageBox.warning(
                self, "Insufficient Data", "Draw a longer curve (at least 10 points)."
            )
            return

        points_array = np.asarray(points, dtype=np.float64)
        x_raw: FloatArray = points_array[:, 0]
        y_raw: FloatArray = points_array[:, 1]

        try:
            preprocessor = StrokePreprocessor(
                self._settings.domain_width, self._settings.domain_height
            )
            x_proc, y_proc = preprocessor.preprocess(x_raw, y_raw)

            if len(x_proc) < 5:
                QMessageBox.warning(
                    self, "Processing Error", "Not enough points after preprocessing."
                )
                return

            is_func, error_msg = preprocessor.is_function(x_proc, y_proc)
            if not is_func:
                QMessageBox.warning(self, "Not a Function", error_msg)
                return
        except (ValueError, TypeError, IndexError) as exc:
            QMessageBox.critical(self, "Error", f"Preprocessing failed: {exc}")
            return

        try:
            models = self._model_service.fit_all_models(
                x_proc, y_proc, self._settings.accuracy
            )
            if not models:
                QMessageBox.warning(
                    self, "Fitting Failed", "Could not fit any model to the data."
                )
                return
            best_model = self._model_service.select_best_model(models, y_proc)
        except (ValueError, RuntimeError, TypeError) as exc:
            QMessageBox.critical(self, "Error", f"Model fitting failed: {exc}")
            return

        self._display_fitted_models(models, best_model, x_proc, y_proc)

    def _display_fitted_models(
        self,
        models: list[FittedModel],
        best_model: FittedModel,
        x_proc: FloatArray,
        y_proc: FloatArray,
    ) -> None:
        y_std = float(np.std(y_proc))
        scores = np.asarray(
            [
                (m.rmse / y_std if y_std > 0 else m.rmse) + 0.1 * m.aic + m.complexity
                for m in models
            ],
            dtype=np.float64,
        )
        sorted_indices = np.argsort(scores)
        top_models = [models[int(i)] for i in sorted_indices[: min(5, len(models))]]

        self._shown_models = top_models

        for curve in self._fitted_curves:
            self._plot_widget.removeItem(curve)
        self._fitted_curves.clear()

        for widget in self._option_widgets:
            widget.setParent(None)  # type: ignore[call-overload]
            widget.deleteLater()
        self._option_widgets.clear()
        self._option_checkboxes.clear()

        latex_parts: list[str] = []
        x_plot = np.linspace(float(np.min(x_proc)), float(np.max(x_proc)), 500, dtype=np.float64)

        for idx, model in enumerate(top_models):
            latex_str = self._latex.generate(model)
            marker = "★ BEST FIT" if model is best_model else ""
            latex_parts.append(
                f"Option {idx + 1} - {model.name} {marker}\nRMSE: {model.rmse:.6f}\n{latex_str}\n"
            )

            try:
                y_plot = np.asarray(model.evaluate(x_plot), dtype=np.float64)
                color = self._COLORS[idx % len(self._COLORS)]
                style = (
                    Qt.PenStyle.SolidLine if model is best_model else Qt.PenStyle.DashLine
                )
                width = 3 if model is best_model else 2

                curve = self._plot_widget.plot(
                    x_plot,
                    y_plot,
                    pen=pg.mkPen(color, width=width, style=style),
                    name=f"Option {idx + 1}: {model.name}",
                )
                self._fitted_curves.append(curve)
            except (ValueError, RuntimeError, OverflowError, FloatingPointError):
                continue

            row, checkbox = self._create_option_row(idx, model.name, marker, color)
            self._options_layout.addWidget(row)
            self._option_widgets.append(row)
            self._option_checkboxes.append(checkbox)

        self._latex_output.setPlainText("\n".join(latex_parts))

    def _create_option_row(
        self, idx: int, model_name: str, marker: str, color: tuple[int, int, int]
    ) -> tuple[QWidget, QCheckBox]:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        checkbox = QCheckBox()
        checkbox.setChecked(True)
        checkbox.toggled.connect(lambda checked, i=idx: self.toggle_option(i, checked))

        r, g, b = color
        label = QLabel(f"Option {idx + 1}: {model_name} {marker}")
        label.setStyleSheet(f"color: rgb({r}, {g}, {b}); font-weight: bold;")

        row_layout.addWidget(checkbox)
        row_layout.addWidget(label)
        row_layout.addStretch(1)

        return row, checkbox

    def toggle_option(self, index: int, checked: bool) -> None:
        if 0 <= index < len(self._fitted_curves):
            self._fitted_curves[index].setVisible(bool(checked))

    def copy_latex(self) -> None:
        latex_text = self._latex_output.toPlainText()
        if latex_text:
            QApplication.clipboard().setText(latex_text)
            QMessageBox.information(self, "Copied", "LaTeX copied to clipboard.")

    def show_settings(self) -> None:
        dialog = SettingsDialog(self._settings, self)
        if dialog.exec():
            new_settings = dialog.get_settings()
            if new_settings is None:
                QMessageBox.critical(self, "Invalid Settings", "Settings values are invalid.")
                return
            try:
                self._settings = new_settings
                self._configure_plot()
                self.clear_drawing()
            except ValueError as exc:
                QMessageBox.critical(self, "Invalid Settings", str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    window = DrawingApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
