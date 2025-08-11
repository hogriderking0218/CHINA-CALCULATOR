import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import math
import cmath
import numpy as np
import sympy as sp
import threading
import requests
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Constants and utilities ---

CONSTANTS = {
    "π": math.pi,
    "e": math.e,
    "φ": (1 + math.sqrt(5)) / 2,
    "γ": 0.5772156649,  # Euler-Mascheroni constant
}

MOTIVATIONAL_QUOTES = [
    "Math is not about numbers, equations, computations, or algorithms: it is about understanding.",
    "Pure mathematics is, in its way, the poetry of logical ideas.",
    "The only way to learn mathematics is to do mathematics.",
    "Mathematics knows no races or geographic boundaries; for mathematics, the cultural world is one country.",
]

MATH_TIPS = [
    "Remember to check units in physics problems.",
    "Draw diagrams for geometry problems to visualize better.",
    "Practice factoring to simplify algebraic expressions.",
    "When stuck, try plugging in numbers to test your solution.",
]

# --- Currency cache and lock ---
currency_cache = {}
currency_lock = threading.Lock()

def fetch_currency_rates():
    url = "https://api.exchangerate-api.com/v4/latest/USD"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            with currency_lock:
                currency_cache.clear()
                currency_cache.update(data["rates"])
        else:
            print("Failed to fetch currency rates")
    except Exception as e:
        print(f"Currency fetch error: {e}")

# --- Helper Functions ---

def safe_eval(expr, variables=None):
    # Evaluate math expressions safely using sympy
    variables = variables or {}
    try:
        expr_sym = sp.sympify(expr)
        return expr_sym.evalf(subs=variables)
    except Exception:
        # fallback to Python eval but restrict globals and locals for safety
        return eval(expr, {"__builtins__": None, "math": math, "cmath": cmath}, variables)

def fraction_to_lowest_terms(num, den):
    gcd = math.gcd(num, den)
    return num // gcd, den // gcd

def poly_roots(coeffs):
    # Use sympy to find roots
    x = sp.symbols('x')
    poly = sum([c * x**i for i, c in enumerate(reversed(coeffs))])
    roots = sp.solve(poly, x)
    return roots

def numeric_diff(func_str, x_val, h=1e-5):
    x = sp.symbols('x')
    f = sp.lambdify(x, sp.sympify(func_str), "math")
    return (f(x_val + h) - f(x_val - h)) / (2 * h)

def numeric_integ(func_str, a, b, n=10000):
    x = sp.symbols('x')
    f = sp.lambdify(x, sp.sympify(func_str), "math")
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return s * h

# --- Main App Class ---

class MiguelCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Miguel's Calculator (Casio fx-991EX ClassWiz Inspired)")
        self.geometry("930x700")
        self.minsize(930, 700)

        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        self.ans = "0"
        self.memory = 0

        self.create_widgets()
        self.create_menu()
        self.set_status("Ready")

        # Fetch currency rates async on startup
        threading.Thread(target=fetch_currency_rates, daemon=True).start()

    def create_menu(self):
        menubar = tk.Menu(self)
        theme_menu = tk.Menu(menubar, tearoff=0)
        theme_menu.add_command(label="Light Mode", command=self.set_light_mode)
        theme_menu.add_command(label="Dark Mode", command=self.set_dark_mode)
        menubar.add_cascade(label="Theme", menu=theme_menu)

        menubar.add_command(label="About", command=self.show_about)
        self.config(menu=menubar)

    def set_light_mode(self):
        self.style.theme_use('clam')
        self.set_status("Light mode activated")

    def set_dark_mode(self):
        self.style.theme_use('alt')
        self.set_status("Dark mode activated")

    def show_about(self):
        messagebox.showinfo("About Miguel's Calculator",
                            "Miguel's Calculator\n"
                            "Inspired by Casio fx-991EX ClassWiz\n"
                            "Author: Your Friendly AI\n"
                            "Features: Normal, Scientific, Fraction, Matrix, Complex, Polynomial, Equation Solver,\n"
                            "Calculus, Graphing, Statistics, Currency & Unit Conversion, Math Practice Quiz, and more!")

    def create_widgets(self):
        # Tabs
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(expand=1, fill="both")

        # Tab frames
        self.tab_normal = ttk.Frame(self.tabs)
        self.tab_scientific = ttk.Frame(self.tabs)
        self.tab_fraction = ttk.Frame(self.tabs)
        self.tab_matrix = ttk.Frame(self.tabs)
        self.tab_complex = ttk.Frame(self.tabs)
        self.tab_poly = ttk.Frame(self.tabs)
        self.tab_equation = ttk.Frame(self.tabs)
        self.tab_calculus = ttk.Frame(self.tabs)
        self.tab_graph = ttk.Frame(self.tabs)
        self.tab_stats = ttk.Frame(self.tabs)
        self.tab_currency = ttk.Frame(self.tabs)
        self.tab_units = ttk.Frame(self.tabs)
        self.tab_practice = ttk.Frame(self.tabs)

        self.tabs.add(self.tab_normal, text="Normal")
        self.tabs.add(self.tab_scientific, text="Scientific")
        self.tabs.add(self.tab_fraction, text="Fraction")
        self.tabs.add(self.tab_matrix, text="Matrix")
        self.tabs.add(self.tab_complex, text="Complex")
        self.tabs.add(self.tab_poly, text="Polynomial")
        self.tabs.add(self.tab_equation, text="Equation Solver")
        self.tabs.add(self.tab_calculus, text="Calculus")
        self.tabs.add(self.tab_graph, text="Graph")
        self.tabs.add(self.tab_stats, text="Statistics")
        self.tabs.add(self.tab_currency, text="Currency")
        self.tabs.add(self.tab_units, text="Units")
        self.tabs.add(self.tab_practice, text="Practice Quiz")

        # Build each tab
        self.build_normal_tab()
        self.build_scientific_tab()
        self.build_fraction_tab()
        self.build_matrix_tab()
        self.build_complex_tab()
        self.build_poly_tab()
        self.build_equation_tab()
        self.build_calculus_tab()
        self.build_graph_tab()
        self.build_stats_tab()
        self.build_currency_tab()
        self.build_units_tab()
        self.build_practice_tab()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        self.status_bar.pack(fill="x", side="bottom")

    def set_status(self, msg):
        self.status_var.set(msg)

    # ---------------- Normal Calculator ----------------
    def build_normal_tab(self):
        frame = self.tab_normal

        self.normal_display = tk.StringVar(value="0")
        display_entry = ttk.Entry(frame, textvariable=self.normal_display, font=("Segoe UI", 20), justify="right")
        display_entry.pack(fill="x", padx=10, pady=10)

        btn_texts = [
            ["7", "8", "9", "÷", "C"],
            ["4", "5", "6", "×", "←"],
            ["1", "2", "3", "-", "="],
            ["0", ".", "+/-", "+", "Ans"],
        ]

        btn_frame = ttk.Frame(frame)
        btn_frame.pack()

        for r, row in enumerate(btn_texts):
            for c, char in enumerate(row):
                b = ttk.Button(btn_frame, text=char, width=5)
                b.grid(row=r, column=c, padx=3, pady=3)
                b.config(command=lambda ch=char: self.normal_btn_click(ch))

    def normal_btn_click(self, char):
        current = self.normal_display.get()

        if char == "C":
            self.normal_display.set("0")
        elif char == "←":
            if len(current) > 1:
                self.normal_display.set(current[:-1])
            else:
                self.normal_display.set("0")
        elif char == "+/-":
            try:
                val = float(current)
                val = -val
                self.normal_display.set(str(val))
            except:
                pass
        elif char == "Ans":
            self.normal_display.set(self.ans)
        elif char == "=":
            expr = current.replace("×", "*").replace("÷", "/")
            try:
                res = safe_eval(expr)
                self.normal_display.set(str(res))
                self.ans = str(res)
                self.set_status("Calculated result")
            except Exception as e:
                messagebox.showerror("Error", f"Invalid expression: {e}")
        else:
            if current == "0":
                self.normal_display.set(char)
            else:
                self.normal_display.set(current + char)

    # ---------------- Scientific Calculator ----------------
    def build_scientific_tab(self):
        frame = self.tab_scientific

        self.scientific_display = tk.StringVar(value="0")
        display_entry = ttk.Entry(frame, textvariable=self.scientific_display, font=("Segoe UI", 20), justify="right")
        display_entry.pack(fill="x", padx=10, pady=10)

        btn_texts = [
            ["sin", "cos", "tan", "log", "ln", "(", ")"],
            ["asin", "acos", "atan", "sinh", "cosh", "tanh", "π"],
            ["7", "8", "9", "×", "÷", "^", "e"],
            ["4", "5", "6", "+", "-", "!", "Ans"],
            ["1", "2", "3", "x²", "x³", "√", "C"],
            ["0", ".", "+/-", "Ans", "=", "M+", "M-"],
            ["MR", "MC", "Ans"]
        ]

        btn_frame = ttk.Frame(frame)
        btn_frame.pack()

        for r, row in enumerate(btn_texts):
            for c, char in enumerate(row):
                b = ttk.Button(btn_frame, text=char, width=5)
                b.grid(row=r, column=c, padx=3, pady=3)
                b.config(command=lambda ch=char: self.scientific_btn_click(ch))

    def scientific_btn_click(self, char):
        cur = self.scientific_display.get()

        def insert_text(t):
            if cur == "0":
                self.scientific_display.set(t)
            else:
                self.scientific_display.set(cur + t)

        try:
            if char in ("sin", "cos", "tan", "asin", "acos", "atan",
                        "sinh", "cosh", "tanh", "log", "ln"):
                insert_text(f"{char}(")
            elif char == "π":
                insert_text("π")
            elif char == "e":
                insert_text("e")
            elif char == "x²":
                self.compute_scientific_power(2)
            elif char == "x³":
                self.compute_scientific_power(3)
            elif char == "√":
                insert_text("sqrt(")
            elif char == "!":
                self.compute_factorial()
            elif char == "C":
                self.scientific_display.set("0")
            elif char == "+/-":
                if cur.startswith("-"):
                    self.scientific_display.set(cur[1:])
                else:
                    self.scientific_display.set("-" + cur)
            elif char == "=":
                expr = self.scientific_display.get()
                expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
                expr = expr.replace("π", str(math.pi)).replace("e", str(math.e))
                try:
                    # Replace math functions with sympy equivalents
                    expr = expr.replace("ln", "log")
                    res = sp.sympify(expr).evalf()
                    self.scientific_display.set(str(res))
                    self.ans = str(res)
                    self.set_status("Calculated result")
                except Exception as e:
                    messagebox.showerror("Error", f"Invalid expression: {e}")
            elif char in ("M+", "M-", "MR", "MC"):
                self.memory_ops(char)
            else:
                insert_text(char)
        except Exception as e:
            messagebox.showerror("Error", f"{e}")

    def compute_scientific_power(self, power):
        try:
            val = float(self.scientific_display.get())
            res = val ** power
            self.scientific_display.set(str(res))
            self.ans = str(res)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input for power: {e}")

    def compute_factorial(self):
        try:
            val = int(float(self.scientific_display.get()))
            if val < 0:
                raise ValueError("Factorial not defined for negative values")
            res = math.factorial(val)
            self.scientific_display.set(str(res))
            self.ans = str(res)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid factorial input: {e}")

    def memory_ops(self, op):
        try:
            val = float(self.scientific_display.get())
        except:
            val = 0
        if op == "M+":
            self.memory += val
            self.set_status("Added to memory")
        elif op == "M-":
            self.memory -= val
            self.set_status("Subtracted from memory")
        elif op == "MR":
            self.scientific_display.set(str(self.memory))
            self.set_status("Memory recalled")
        elif op == "MC":
            self.memory = 0
            self.set_status("Memory cleared")

    # ---------------- Fraction Calculator ----------------
    def build_fraction_tab(self):
        frame = self.tab_fraction

        # Input for fraction1
        f1_frame = ttk.LabelFrame(frame, text="Fraction 1")
        f1_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(f1_frame, text="Numerator:").grid(row=0, column=0)
        self.frac1_num = ttk.Entry(f1_frame, width=8)
        self.frac1_num.grid(row=0, column=1, padx=5)
        ttk.Label(f1_frame, text="Denominator:").grid(row=0, column=2)
        self.frac1_den = ttk.Entry(f1_frame, width=8)
        self.frac1_den.grid(row=0, column=3, padx=5)

        # Input for fraction2
        f2_frame = ttk.LabelFrame(frame, text="Fraction 2")
        f2_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(f2_frame, text="Numerator:").grid(row=0, column=0)
        self.frac2_num = ttk.Entry(f2_frame, width=8)
        self.frac2_num.grid(row=0, column=1, padx=5)
        ttk.Label(f2_frame, text="Denominator:").grid(row=0, column=2)
        self.frac2_den = ttk.Entry(f2_frame, width=8)
        self.frac2_den.grid(row=0, column=3, padx=5)

        # Operation selector
        op_frame = ttk.Frame(frame)
        op_frame.pack(pady=10)
        ttk.Label(op_frame, text="Operation:").pack(side="left")
        self.frac_op = tk.StringVar(value="+")
        ops = ["+", "-", "*", "/"]
        for op in ops:
            ttk.Radiobutton(op_frame, text=op, value=op, variable=self.frac_op).pack(side="left", padx=5)

        # Calculate button
        ttk.Button(frame, text="Calculate", command=self.calculate_fraction).pack(pady=5)

        # Result display
        self.frac_result_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.frac_result_var, font=("Segoe UI", 14, "bold")).pack(pady=5)

    def calculate_fraction(self):
        try:
            n1 = int(self.frac1_num.get())
            d1 = int(self.frac1_den.get())
            n2 = int(self.frac2_num.get())
            d2 = int(self.frac2_den.get())
            if d1 == 0 or d2 == 0:
                raise ValueError("Denominator cannot be zero")
            op = self.frac_op.get()

            # Convert to fractions
            f1 = sp.Rational(n1, d1)
            f2 = sp.Rational(n2, d2)

            if op == "+":
                res = f1 + f2
            elif op == "-":
                res = f1 - f2
            elif op == "*":
                res = f1 * f2
            else:
                res = f1 / f2

            # Simplify
            res = res.cancel()

            self.frac_result_var.set(f"Result: {res} (≈ {float(res):.6g})")
            self.set_status("Fraction calculated")

        except Exception as e:
            messagebox.showerror("Error", f"Invalid fraction input: {e}")

    # ---------------- Matrix Calculator ----------------
    def build_matrix_tab(self):
        frame = self.tab_matrix

        ttk.Label(frame, text="Matrix A (2x2)", font=("Segoe UI", 12, "bold")).pack(pady=5)
        self.matA_entries = []
        matA_frame = ttk.Frame(frame)
        matA_frame.pack()
        for r in range(2):
            row_entries = []
            for c in range(2):
                e = ttk.Entry(matA_frame, width=5, justify="center")
                e.grid(row=r, column=c, padx=3, pady=3)
                row_entries.append(e)
            self.matA_entries.append(row_entries)

        ttk.Label(frame, text="Matrix B (2x2)", font=("Segoe UI", 12, "bold")).pack(pady=5)
        self.matB_entries = []
        matB_frame = ttk.Frame(frame)
        matB_frame.pack()
        for r in range(2):
            row_entries = []
            for c in range(2):
                e = ttk.Entry(matB_frame, width=5, justify="center")
                e.grid(row=r, column=c, padx=3, pady=3)
                row_entries.append(e)
            self.matB_entries.append(row_entries)

        op_frame = ttk.Frame(frame)
        op_frame.pack(pady=10)
        ttk.Label(op_frame, text="Operation:").pack(side="left")
        self.matrix_op = tk.StringVar(value="Add")
        ops = ["Add", "Multiply"]
        for op in ops:
            ttk.Radiobutton(op_frame, text=op, value=op, variable=self.matrix_op).pack(side="left", padx=5)

        ttk.Button(frame, text="Calculate", command=self.calculate_matrix).pack(pady=5)

        self.matrix_result_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.matrix_result_var, font=("Segoe UI", 12, "bold")).pack(pady=5)

    def read_matrix_entries(self, entries):
        try:
            mat = []
            for row in entries:
                rvals = []
                for e in row:
                    val = float(e.get())
                    rvals.append(val)
                mat.append(rvals)
            return np.array(mat)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid matrix input: {e}")
            return None

    def calculate_matrix(self):
        A = self.read_matrix_entries(self.matA_entries)
        B = self.read_matrix_entries(self.matB_entries)
        if A is None or B is None:
            return

        op = self.matrix_op.get()
        try:
            if op == "Add":
                res = A + B
            else:
                res = np.dot(A, B)
            res_str = "\n".join(["\t".join(f"{val:.2f}" for val in row) for row in res])
            self.matrix_result_var.set(f"Result:\n{res_str}")
            self.set_status("Matrix operation done")
        except Exception as e:
            messagebox.showerror("Error", f"Matrix calculation error: {e}")

    # ---------------- Complex Numbers ----------------
    def build_complex_tab(self):
        frame = self.tab_complex

        ttk.Label(frame, text="Real Part:", font=("Segoe UI", 12)).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(frame, text="Imaginary Part:", font=("Segoe UI", 12)).grid(row=1, column=0, padx=5, pady=5)

        self.complex_real = ttk.Entry(frame, width=15)
        self.complex_imag = ttk.Entry(frame, width=15)
        self.complex_real.grid(row=0, column=1, padx=5, pady=5)
        self.complex_imag.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Convert to Polar", command=self.convert_complex_to_polar).grid(row=2, column=0, columnspan=2, pady=5)

        ttk.Label(frame, text="Magnitude (r):", font=("Segoe UI", 12)).grid(row=3, column=0, padx=5, pady=5)
        ttk.Label(frame, text="Angle (θ in degrees):", font=("Segoe UI", 12)).grid(row=4, column=0, padx=5, pady=5)

        self.complex_r_var = tk.StringVar(value="")
        self.complex_theta_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.complex_r_var, font=("Segoe UI", 12, "bold")).grid(row=3, column=1)
        ttk.Label(frame, textvariable=self.complex_theta_var, font=("Segoe UI", 12, "bold")).grid(row=4, column=1)

        ttk.Separator(frame, orient="horizontal").grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)

        ttk.Label(frame, text="Magnitude (r):", font=("Segoe UI", 12)).grid(row=6, column=0, padx=5, pady=5)
        ttk.Label(frame, text="Angle (θ in degrees):", font=("Segoe UI", 12)).grid(row=7, column=0, padx=5, pady=5)

        self.polar_r = ttk.Entry(frame, width=15)
        self.polar_theta = ttk.Entry(frame, width=15)
        self.polar_r.grid(row=6, column=1, padx=5, pady=5)
        self.polar_theta.grid(row=7, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Convert to Rectangular", command=self.convert_polar_to_complex).grid(row=8, column=0, columnspan=2, pady=5)

        self.complex_rect_var = tk.StringVar(value="")
        ttk.Label(frame, text="Rectangular form:", font=("Segoe UI", 12)).grid(row=9, column=0, padx=5, pady=5)
        ttk.Label(frame, textvariable=self.complex_rect_var, font=("Segoe UI", 12, "bold")).grid(row=9, column=1)

    def convert_complex_to_polar(self):
        try:
            r = float(self.complex_real.get())
            i = float(self.complex_imag.get())
            c = complex(r, i)
            mag = abs(c)
            angle = math.degrees(cmath.phase(c))
            self.complex_r_var.set(f"{mag:.5g}")
            self.complex_theta_var.set(f"{angle:.5g}")
            self.set_status("Converted complex to polar")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid complex input: {e}")

    def convert_polar_to_complex(self):
        try:
            r = float(self.polar_r.get())
            theta_deg = float(self.polar_theta.get())
            theta_rad = math.radians(theta_deg)
            c = cmath.rect(r, theta_rad)
            self.complex_rect_var.set(f"{c.real:.5g} + {c.imag:.5g}i")
            self.set_status("Converted polar to complex")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid polar input: {e}")

    # ---------------- Polynomial Solver ----------------
    def build_poly_tab(self):
        frame = self.tab_poly

        ttk.Label(frame, text="Enter coefficients (highest degree first), comma separated:", font=("Segoe UI", 12)).pack(pady=5)
        self.poly_coeffs_entry = ttk.Entry(frame, width=60)
        self.poly_coeffs_entry.pack(pady=5)

        ttk.Button(frame, text="Find Roots", command=self.find_poly_roots).pack(pady=5)

        self.poly_roots_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.poly_roots_var, font=("Segoe UI", 12, "bold")).pack(pady=5)

    def find_poly_roots(self):
        coeffs_text = self.poly_coeffs_entry.get()
        try:
            coeffs = [float(c) for c in coeffs_text.split(",")]
            roots = poly_roots(coeffs)
            roots_str = ", ".join([str(r.evalf()) for r in roots])
            self.poly_roots_var.set(f"Roots: {roots_str}")
            self.set_status("Polynomial roots found")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid polynomial coefficients: {e}")

    # ---------------- Equation Solver ----------------
    def build_equation_tab(self):
        frame = self.tab_equation

        ttk.Label(frame, text="Solve linear or quadratic equation ax² + bx + c = 0", font=("Segoe UI", 12)).pack(pady=5)

        coef_frame = ttk.Frame(frame)
        coef_frame.pack(pady=5)
        ttk.Label(coef_frame, text="a:").grid(row=0, column=0, padx=5)
        ttk.Label(coef_frame, text="b:").grid(row=0, column=2, padx=5)
        ttk.Label(coef_frame, text="c:").grid(row=0, column=4, padx=5)

        self.eq_a = ttk.Entry(coef_frame, width=8)
        self.eq_b = ttk.Entry(coef_frame, width=8)
        self.eq_c = ttk.Entry(coef_frame, width=8)
        self.eq_a.grid(row=0, column=1, padx=5)
        self.eq_b.grid(row=0, column=3, padx=5)
        self.eq_c.grid(row=0, column=5, padx=5)

        ttk.Button(frame, text="Solve", command=self.solve_equation).pack(pady=5)

        self.eq_result_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.eq_result_var, font=("Segoe UI", 12, "bold")).pack(pady=5)

    def solve_equation(self):
        try:
            a = float(self.eq_a.get())
            b = float(self.eq_b.get())
            c = float(self.eq_c.get())
            x = sp.symbols('x')
            eq = a * x**2 + b * x + c
            roots = sp.solve(eq, x)
            roots_str = ", ".join([str(r.evalf()) for r in roots])
            self.eq_result_var.set(f"Roots: {roots_str}")
            self.set_status("Equation solved")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    # ---------------- Calculus ----------------
    def build_calculus_tab(self):
        frame = self.tab_calculus

        ttk.Label(frame, text="Enter function f(x):", font=("Segoe UI", 12)).pack(pady=5)
        self.calc_func_entry = ttk.Entry(frame, width=50)
        self.calc_func_entry.pack(pady=5)

        diff_btn = ttk.Button(frame, text="Differentiate at x", command=self.calculus_diff)
        diff_btn.pack(pady=5)

        integ_btn = ttk.Button(frame, text="Integrate between a and b", command=self.calculus_integ)
        integ_btn.pack(pady=5)

        ttk.Label(frame, text="x / a / b values (comma separated):").pack(pady=5)
        self.calc_vals_entry = ttk.Entry(frame, width=30)
        self.calc_vals_entry.pack(pady=5)

        self.calc_result_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.calc_result_var, font=("Segoe UI", 12, "bold")).pack(pady=5)

    def calculus_diff(self):
        func = self.calc_func_entry.get()
        vals_text = self.calc_vals_entry.get()
        try:
            x_val = float(vals_text.split(",")[0])
            res = numeric_diff(func, x_val)
            self.calc_result_var.set(f"f'({x_val}) = {res:.6g}")
            self.set_status("Differentiation done")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input for differentiation: {e}")

    def calculus_integ(self):
        func = self.calc_func_entry.get()
        vals_text = self.calc_vals_entry.get()
        try:
            a, b = map(float, vals_text.split(",")[1:3])
            res = numeric_integ(func, a, b)
            self.calc_result_var.set(f"∫ f(x) dx from {a} to {b} = {res:.6g}")
            self.set_status("Integration done")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input for integration: {e}")

    # ---------------- Graphing ----------------
    def build_graph_tab(self):
        frame = self.tab_graph

        ttk.Label(frame, text="Enter function f(x):", font=("Segoe UI", 12)).pack(pady=5)
        self.graph_func_entry = ttk.Entry(frame, width=60)
        self.graph_func_entry.pack(pady=5)

        ttk.Label(frame, text="x range (start,end):").pack(pady=5)
        self.graph_range_entry = ttk.Entry(frame, width=30)
        self.graph_range_entry.pack(pady=5)

        ttk.Button(frame, text="Plot Graph", command=self.plot_graph).pack(pady=5)

        self.graph_canvas_frame = ttk.Frame(frame)
        self.graph_canvas_frame.pack(expand=1, fill="both")

    def plot_graph(self):
        func_str = self.graph_func_entry.get()
        try:
            start, end = map(float, self.graph_range_entry.get().split(","))
            x = sp.symbols('x')
            f = sp.lambdify(x, sp.sympify(func_str), "numpy")
            xs = np.linspace(start, end, 400)
            ys = f(xs)

            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            ax.plot(xs, ys)
            ax.set_title(f"Graph of f(x) = {func_str}")
            ax.grid(True)

            for widget in self.graph_canvas_frame.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(fig, master=self.graph_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=1)
            self.set_status("Graph plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Graph plotting failed: {e}")

    # ---------------- Statistics ----------------
    def build_stats_tab(self):
        frame = self.tab_stats

        ttk.Label(frame, text="Enter numbers separated by commas:", font=("Segoe UI", 12)).pack(pady=5)
        self.stats_data_entry = ttk.Entry(frame, width=60)
        self.stats_data_entry.pack(pady=5)

        ttk.Button(frame, text="Calculate Statistics", command=self.calculate_stats).pack(pady=5)

        self.stats_result_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.stats_result_var, font=("Segoe UI", 12, "bold")).pack(pady=5)

    def calculate_stats(self):
        data_text = self.stats_data_entry.get()
        try:
            nums = list(map(float, data_text.split(",")))
            mean = np.mean(nums)
            median = np.median(nums)
            mode = None
            try:
                mode = max(set(nums), key=nums.count)
            except:
                mode = "N/A"
            std_dev = np.std(nums)
            variance = np.var(nums)

            res = (f"Mean: {mean:.6g}\nMedian: {median:.6g}\nMode: {mode}\n"
                   f"Standard Deviation: {std_dev:.6g}\nVariance: {variance:.6g}")
            self.stats_result_var.set(res)
            self.set_status("Statistics calculated")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid data input: {e}")

    # ---------------- Currency Converter ----------------
    def build_currency_tab(self):
        frame = self.tab_currency

        ttk.Label(frame, text="Amount:", font=("Segoe UI", 12)).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(frame, text="From (Currency Code):", font=("Segoe UI", 12)).grid(row=1, column=0, padx=5, pady=5)
        ttk.Label(frame, text="To (Currency Code):", font=("Segoe UI", 12)).grid(row=2, column=0, padx=5, pady=5)

        self.currency_amount = ttk.Entry(frame, width=20)
        self.currency_from = ttk.Entry(frame, width=20)
        self.currency_to = ttk.Entry(frame, width=20)

        self.currency_amount.grid(row=0, column=1, padx=5, pady=5)
        self.currency_from.grid(row=1, column=1, padx=5, pady=5)
        self.currency_to.grid(row=2, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Convert", command=self.convert_currency).grid(row=3, column=0, columnspan=2, pady=10)

        self.currency_result_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.currency_result_var, font=("Segoe UI", 12, "bold")).grid(row=4, column=0, columnspan=2)

        ttk.Button(frame, text="Refresh Rates", command=self.refresh_currency_rates).grid(row=5, column=0, columnspan=2)

    def convert_currency(self):
        try:
            amt = float(self.currency_amount.get())
            frm = self.currency_from.get().strip().upper()
            to = self.currency_to.get().strip().upper()

            with currency_lock:
                if not currency_cache:
                    messagebox.showwarning("Warning", "Currency rates not loaded yet. Please refresh.")
                    return
                rates = currency_cache

            if frm not in rates or to not in rates:
                messagebox.showerror("Error", "Currency code not found.")
                return

            usd_amt = amt / rates[frm]
            converted = usd_amt * rates[to]

            self.currency_result_var.set(f"{amt} {frm} = {converted:.4f} {to}")
            self.set_status("Currency converted")
        except Exception as e:
            messagebox.showerror("Error", f"Conversion failed: {e}")

    def refresh_currency_rates(self):
        threading.Thread(target=fetch_currency_rates, daemon=True).start()
        self.set_status("Refreshing currency rates...")

    # ---------------- Unit Conversion ----------------
    def build_units_tab(self):
        frame = self.tab_units

        ttk.Label(frame, text="Value:", font=("Segoe UI", 12)).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(frame, text="From Unit:", font=("Segoe UI", 12)).grid(row=1, column=0, padx=5, pady=5)
        ttk.Label(frame, text="To Unit:", font=("Segoe UI", 12)).grid(row=2, column=0, padx=5, pady=5)

        self.unit_value = ttk.Entry(frame, width=20)
        self.unit_from = ttk.Combobox(frame, width=18, state="readonly")
        self.unit_to = ttk.Combobox(frame, width=18, state="readonly")

        self.unit_value.grid(row=0, column=1, padx=5, pady=5)
        self.unit_from.grid(row=1, column=1, padx=5, pady=5)
        self.unit_to.grid(row=2, column=1, padx=5, pady=5)

        units = {
            "Length": {"m": 1, "cm": 0.01, "mm": 0.001, "km": 1000, "inch": 0.0254, "ft": 0.3048, "yard": 0.9144, "mile": 1609.34},
            "Weight": {"kg": 1, "g": 0.001, "mg": 1e-6, "lb": 0.453592, "oz": 0.0283495},
            "Temperature": {"C": ("C", "temp"), "F": ("F", "temp"), "K": ("K", "temp")},
            "Time": {"s": 1, "min": 60, "hr": 3600, "day": 86400}
        }
        self.units_data = units

        self.unit_categories = list(units.keys())
        self.current_unit_category = tk.StringVar()
        ttk.Label(frame, text="Category:").grid(row=3, column=0, padx=5, pady=5)
        self.category_combo = ttk.Combobox(frame, values=self.unit_categories, state="readonly", width=18)
        self.category_combo.grid(row=3, column=1, padx=5, pady=5)
        self.category_combo.bind("<<ComboboxSelected>>", self.update_units)

        ttk.Button(frame, text="Convert", command=self.convert_units).grid(row=4, column=0, columnspan=2, pady=10)

        self.unit_result_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.unit_result_var, font=("Segoe UI", 12, "bold")).grid(row=5, column=0, columnspan=2)

        self.category_combo.current(0)
        self.update_units()

    def update_units(self, event=None):
        category = self.category_combo.get()
        if not category:
            return
        units = list(self.units_data[category].keys())
        self.unit_from.config(values=units)
        self.unit_to.config(values=units)
        if units:
            self.unit_from.current(0)
            self.unit_to.current(0)

    def convert_units(self):
        try:
            val = float(self.unit_value.get())
            frm = self.unit_from.get()
            to = self.unit_to.get()
            cat = self.category_combo.get()

            if cat == "Temperature":
                # special conversion
                res = self.convert_temperature(val, frm, to)
            else:
                base_val = val * self.units_data[cat][frm]
                res = base_val / self.units_data[cat][to]

            self.unit_result_var.set(f"{val} {frm} = {res:.6g} {to}")
            self.set_status("Unit converted")
        except Exception as e:
            messagebox.showerror("Error", f"Unit conversion failed: {e}")

    def convert_temperature(self, val, frm, to):
        # Convert between Celsius, Fahrenheit, Kelvin
        if frm == to:
            return val
        # Convert frm to Celsius
        if frm == "C":
            c = val
        elif frm == "F":
            c = (val - 32) * 5 / 9
        elif frm == "K":
            c = val - 273.15
        else:
            raise ValueError("Unknown temperature unit")

        # Celsius to target
        if to == "C":
            return c
        elif to == "F":
            return c * 9 / 5 + 32
        elif to == "K":
            return c + 273.15
        else:
            raise ValueError("Unknown temperature unit")

    # ---------------- Practice Quiz ----------------
    def build_practice_tab(self):
        frame = self.tab_practice

        self.quiz_score = 0
        self.quiz_total = 0

        self.quiz_question_var = tk.StringVar(value="Press 'Next Question' to start")
        ttk.Label(frame, textvariable=self.quiz_question_var, font=("Segoe UI", 14)).pack(pady=10)

        self.quiz_answer_entry = ttk.Entry(frame, width=20, font=("Segoe UI", 14))
        self.quiz_answer_entry.pack(pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Next Question", command=self.next_quiz_question).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Submit Answer", command=self.submit_quiz_answer).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Reset Score", command=self.reset_quiz_score).pack(side="left", padx=5)

        self.quiz_feedback_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.quiz_feedback_var, font=("Segoe UI", 12)).pack(pady=5)

        self.quiz_score_var = tk.StringVar(value="Score: 0 / 0")
        ttk.Label(frame, textvariable=self.quiz_score_var, font=("Segoe UI", 12, "bold")).pack(pady=5)

        self.current_quiz_question = None
        self.current_quiz_answer = None

    def generate_quiz_question(self):
        # For demo, generate simple fraction add/subtract
        ops = ["+", "-"]
        op = random.choice(ops)
        n1 = random.randint(1, 10)
        d1 = random.randint(1, 10)
        n2 = random.randint(1, 10)
        d2 = random.randint(1, 10)

        q = f"Solve: {n1}/{d1} {op} {n2}/{d2}"
        f1 = sp.Rational(n1, d1)
        f2 = sp.Rational(n2, d2)

        if op == "+":
            ans = f1 + f2
        else:
            ans = f1 - f2

        return q, ans

    def next_quiz_question(self):
        q, ans = self.generate_quiz_question()
        self.current_quiz_question = q
        self.current_quiz_answer = ans
        self.quiz_question_var.set(q)
        self.quiz_answer_entry.delete(0, tk.END)
        self.quiz_feedback_var.set("")
        self.set_status("New quiz question")

    def submit_quiz_answer(self):
        user_ans = self.quiz_answer_entry.get()
        try:
            # Try to parse user answer as fraction or float
            if "/" in user_ans:
                num, den = user_ans.split("/")
                user_frac = sp.Rational(int(num.strip()), int(den.strip()))
            else:
                user_frac = sp.Rational(float(user_ans))

            correct = (user_frac == self.current_quiz_answer)
            self.quiz_total += 1
            if correct:
                self.quiz_score += 1
                self.quiz_feedback_var.set("Correct!")
            else:
                self.quiz_feedback_var.set(f"Incorrect! Correct answer: {self.current_quiz_answer}")

            self.quiz_score_var.set(f"Score: {self.quiz_score} / {self.quiz_total}")
            self.set_status("Quiz answered")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid answer format: {e}")

    def reset_quiz_score(self):
        self.quiz_score = 0
        self.quiz_total = 0
        self.quiz_score_var.set("Score: 0 / 0")
        self.quiz_feedback_var.set("")
        self.set_status("Quiz score reset")

# --- Run the app ---

if __name__ == "__main__":
    app = MiguelCalculator()
    app.mainloop()
