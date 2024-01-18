import math  # for gamma
import streamlit as st
import sympy as sm
import matplotlib.pyplot as plt
import numpy as np
import re
from streamlit_option_menu import option_menu
from sympy import ceiling


def gamma(n):
    if n.is_integer():
        # Calculate the gamma function of an integer
        result = math.gamma(int(n))
    elif n == (0.5):
        result = "\sqrt{\pi}"  #
    else:
        # Calculate the gamma function of a float
        result = math.gamma(round(n, 3))
    return result

def reimann_liovelle_for_constant_dec(f, alpha):  # alpha being decimal
    x = sm.Symbol('x')
    a = sm.Symbol('a')
    t = sm.Symbol('t')
    c = sm.Symbol('c')
    alp = sm.Symbol('α')
    numbers = set(str(i) for i in range(1, 1000))
    n = [1, 2, 3, 4, ..., 1000]
    decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    if (f == f"x^{numbers}") and (alpha in decimal):
        st.latex("Hello")

    if (f in numbers) and (alpha in decimal):
        st.markdown(f"##### Using Reimann-Liouvelle integral from $'a'$ to $'x'$ to the order ${alp}$, we get, #####")
        st.latex(rf"_{{{a}}}I_{{{x}}}^ α(f) = \frac{{ {c}({x - a})^{alp} }}{{ {alp}\Gamma({alp})}} ")
        st.markdown(f"##### Here, ${alp}={alpha}$ and when ${a}={0}$,")

        st.latex(
            rf"_{{{0}}}I_{{{x}}}^ {{{alpha}}}({f}) = \frac{{ {f}×({x})^{ {alpha} } }} {{ {alpha}×{gamma(alpha)} }}")
        st.markdown("##### Plotting of Reimann-Liouvelle integral: #####")

        if alpha == 0.5:
            pi = 3.14159265359
            plt.style.use(['tableau-colorblind10', 'ggplot'])

            x = np.linspace(0, 1, 100)

            y = (float(f) * x ** alpha) / (alpha * gamma(pi))

            plt.plot(x, y)
            plt.xlabel("g(x)")
            plt.ylabel("f(x)")
            plt.title(f"g(x) = {f} * x^{alpha} / ({alpha} * gamma({alpha}))")

            fig = plt.gcf()
            st.pyplot(fig)
        else:
            plt.style.use(['tableau-colorblind10', 'ggplot'])

            x = np.linspace(0, 1, 100)

            y = (float(f) * x ** alpha) / (alpha * gamma(alpha))

            plt.plot(x, y)
            plt.xlabel("g(x)")
            plt.ylabel("f(x)")
            plt.title(f"g(x) = {f} * x^{alpha} / ({alpha} * gamma({alpha}))")

            fig = plt.gcf()
            st.pyplot(fig)


def normal_integration(f, alpha):  # alpha being equal to 1
    x = sm.Symbol('x')
    a = sm.Symbol('a')
    t = sm.Symbol('t')
    s = sm.Symbol('s')
    c = sm.Symbol('c')
    alp = sm.Symbol('α')
    numbers = (int(i) for i in range(1, 1000))
    decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    if (f == ""):
        st.markdown("<h5>Type any function to get the answer...</h5>", unsafe_allow_html=True)

    else:
        x = sm.Symbol('x')
        a = sm.Symbol('a')
        t = sm.Symbol('t')
        numbers = list(str(i) for i in range(1, 1000))
        decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        st.markdown(
            f"##### Using Reimann-Liouvelle integral from $'a'$ to $'x'$ to the order $α = {alpha}$, we get, #####")
        st.latex(rf"_{{{a}}}I_{{{x}}}^ α(f) = \frac{{ {c}({x - a})^{alp} }}{{ {alp}\Gamma({alp})}} ")
        st.latex(rf"=\frac{{ {f}×({x - a})^{ {alpha} } }} {{ {alpha}×{gamma(alpha)} }}")
        st.markdown(f"##### Here, ${alp}={alpha}$ and when ${a}={0}$,")
        st.latex(rf"=\frac{{ {f}×({x})^{ {alpha} } }} {{ {alpha}×{gamma(alpha)} }}")

        x = sm.Symbol('x')
        st.latex(f"= {f}{x}")

        f = float(f)
        g = lambda x: f * x

        st.markdown("##### Plotting of Reimann-Liouvelle integral: #####")
        plt.style.use(['tableau-colorblind10', 'ggplot'])

        x = np.linspace(0, 1, 100)
        y = g(x)

        plt.plot(x, y)
        plt.xlabel("g(x)")
        plt.ylabel("f(x)")
        plt.title(f"g(x) = {f}x")
        fig = plt.gcf()
        st.pyplot(fig)

def floating_coeff_x_pow(f, alpha):
    x = sm.Symbol('x')
    a = sm.Symbol('a')
    t = sm.Symbol('t')
    s = sm.Symbol('s')
    n = sm.Symbol('n')
    α = sm.Symbol('α')

    fraccoef = ["{:.1f}".format(num) for num in np.arange(1.0, 100.1, 0.1)]
    fraccoef_1 = [float(num) for num in fraccoef]
    fracpow = ["{:.1f}".format(num) for num in np.arange(1.0, 100.1, 0.1)]
    fracpow_1 = [float(num) for num in fracpow]

    decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]

    for fp in fracpow_1:
        for fc in fraccoef_1:
            if f == f"{fc}*x^{fp}" and alpha in decimal:
                st.markdown(
                    f"##### Using Reimann-Liovelle integral from $'a'$ to $'x'$ to the order $α$, we get, #####")
                st.latex(
                    rf"_{{{a}}}I_{{{x}}}^ α(f) =c × \frac{{ {x}^{{{n + α}}} }}{{ {{\Gamma({α})}} }}×\frac {{\Gamma({α})×\Gamma({n + 1})}} {{\Gamma({α + n + 1}) }}")
                st.markdown(f"##### Here, ${α}={alpha}$ and when ${a}={0}$,")
                st.latex(
                    rf"_{{{0}}}I_{{{x}}}^ {{{alpha}}}({fc}x^{{{fp}}}) = {fc} × \frac{{{x}^{{{round(alpha - 1 + fp + 1, 2)}}}}} {{\Gamma({alpha})}} ×\frac{{ \Gamma{round(alpha - 1 + 1, 2)}×\Gamma{fp + 1} }}{{ \Gamma{fp + alpha + 1} }}")
                st.latex(
                    rf"={fc} × \frac{{ {x}^{{{fp + alpha}}}×{round(gamma(fp + 1.0), 2)} }} {{ {round(gamma(fp + alpha + 1.0), 3)} }}")
                st.latex(
                    rf"= \frac{{ {x}^{{{fp + alpha}}}×{round(fc * (gamma(fp + 1.0)), 2)} }} {{ {round(gamma(fp + alpha + 1.0), 3)} }}")

                st.markdown("##### Plotting of Reimann-Liouvelle integral: #####")
                plt.style.use(['tableau-colorblind10', 'ggplot'])

                x = np.linspace(0, 1, 100)

                y = (x ** (fp + alpha) * gamma(fp + 1.0)) / round(gamma(fp + alpha + 1.0), 3)

                plt.plot(x, y)
                plt.xlabel("g(x)")
                plt.ylabel("f(x)")
                plt.title(
                    rf"g(x) = x^({fp + alpha})×{round(fc * (gamma(fp + 1.0)), 2)} / {round(gamma(fp + alpha + 1.0), 3)} ")
                fig = plt.gcf()
                st.pyplot(fig)

def floating_x_power(f, alpha):  # of the form : x^4.5
    x = sm.Symbol('x')
    a = sm.Symbol('a')
    t = sm.Symbol('t')
    s = sm.Symbol('s')
    n = sm.Symbol('n')
    α = sm.Symbol('α')
    decpow_1 = ["{:.1f}".format(num) for num in np.arange(1.0, 1000.1, 0.1)]
    decpow = [float(num) for num in decpow_1]

    decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]

    for dec in decpow:
        if f == f"x^{dec}" and alpha in decimal:
            st.markdown(
                f"##### Using Reimann-Liovelle integral from $'a'$ to $'x'$ to the order $α$, we get, #####")
            st.latex(
                rf"_{{{a}}}I_{{{x}}}^ α(f) =\frac{{ {x}^{{{n + α}}} }}{{ {{\Gamma({α})}} }}×\frac {{\Gamma({α})×\Gamma({n + 1})}} {{\Gamma({α + n + 1}) }}")
            st.markdown(f"##### Here, ${α}={alpha}$ and when ${a}={0}$,")
            st.latex(
                rf"_{{{0}}}I_{{{x}}}^ {{{alpha}}}({f}) = \frac{{{x}^{{{round(alpha - 1.0 + dec + 1.0, 2)}}}}} {{\Gamma({alpha})}} ×\frac{{ \Gamma{round(alpha - 1 + 1, 2)}×\Gamma{dec + 1} }}{{ \Gamma{dec + alpha + 1} }}")
            st.latex(
                rf"=\frac{{ {x}^{{{dec + alpha}}}×{round(gamma(dec + 1.0), 3)} }} {{ {round(gamma(dec + alpha + 1.0), 3)} }}")

            st.markdown("##### Plotting of Reimann-Liouvelle integral: #####")

            plt.style.use(['tableau-colorblind10', 'ggplot'])

            x = np.linspace(0, 1, 100)

            y = (x ** (dec + alpha) * gamma(dec + 1.0)) / round(gamma(dec + alpha + 1.0), 3)

            plt.plot(x, y)
            plt.xlabel("g(x)")
            plt.ylabel("f(x)")
            plt.title(
                f"g(x) = x^({dec + alpha}) * gamma({dec + 1.0}) / gamma({dec + alpha + 1.0}) rounded to 3 decimal places")

            fig = plt.gcf()
            st.pyplot(fig)
            break

def reimann_liovelle_for_variable_dec(f, alpha):
    x = sm.Symbol('x')
    a = sm.Symbol('a')
    t = sm.Symbol('t')
    s = sm.Symbol('s')
    n = sm.Symbol('n')
    α = sm.Symbol('α')
    numbers = (int(i) for i in range(1, 1000))
    decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]

    for number in numbers:
        if f == f"x^{number}" and alpha in decimal:
            st.markdown(
                f"##### Using Reimann-Liovelle integral from $'a'$ to $'x'$ to the order $α$, we get, #####")
            st.latex(
                rf"_{{{a}}}I_{{{x}}}^ α(f) =\frac{{ {x}^{{{n + α}}} }}{{ {{\Gamma({α})}} }}×\frac {{\Gamma({α})×\Gamma({n + 1})}} {{\Gamma({α + n + 1}) }}")
            st.markdown(f"##### Here, ${α}={alpha}$ and when ${a}={0}$,")
            st.latex(
                rf"_{{{0}}}I_{{{x}}}^ {{{alpha}}}({f}) = \frac{{{x}^{{{round(alpha - 1 + number + 1, 2)}}}}} {{\Gamma({alpha})}} ×\frac{{ \Gamma{round(alpha - 1 + 1, 2)}×\Gamma{number + 1} }}{{ \Gamma{number + alpha + 1} }}")
            st.latex(
                rf"=\frac{{ {x}^{{{number + alpha}}}×{gamma(number + 1.0)} }} {{ {round(gamma(number + alpha + 1.0), 3)} }}")

            st.markdown("##### Plotting of Reimann-Liouvelle integral: #####")

            plt.style.use(['tableau-colorblind10', 'ggplot'])

            x = np.linspace(0, 1, 100)

            y = (x ** (number + alpha) * gamma(number + 1.0)) / round(gamma(number + alpha + 1.0), 3)

            plt.plot(x, y)
            plt.xlabel("g(x)")
            plt.ylabel("f(x)")
            plt.title(
                f"g(x) = x^({number + alpha}) * gamma({number + 1.0}) / gamma({number + alpha + 1.0}) rounded to 3 decimal places")

            fig = plt.gcf()
            st.pyplot(fig)
            break

def normal_integration_var(f, alpha):
    x = sm.Symbol('x')
    a = sm.Symbol('a')
    t = sm.Symbol('t')
    s = sm.Symbol('s')
    n = sm.Symbol('n')
    α = sm.Symbol('α')

    numbers = (int(i) for i in range(0, 1000))
    decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]

    for number in numbers:
        if f == f"x^{number}" and alpha == 1:
            st.markdown(
                f"##### Using Reimann-Liouvelle integral from $'a'$ to $'x'$ to the order $α$, we get, #####")
            st.latex(
                rf"_{{{a}}}I_{{{x}}}^ α(f) =\frac{{ {x}^{{{n + α}}} }}{{ {{\Gamma({α})}} }}×\frac {{\Gamma({α})×\Gamma({n + 1})}} {{\Gamma({α + n + 1}) }}")
            st.markdown(f"##### Here, ${α}={alpha}$ and when ${a}={0}$,")
            st.latex(
                rf"_{{{0}}}I_{{{x}}}^ {{{alpha}}}({f}) = \frac{{{x}^{{{round(alpha - 1 + number + 1, 2)}}}}} {{\Gamma({alpha})}} \frac{{ \Gamma{round(alpha - 1 + 1, 2)}×\Gamma{number + 1} }}{{ \Gamma{number + alpha + 1} }}")
            st.latex(
                rf"=\frac{{ {x}^{{{number + alpha}}}×{gamma(number + 1.0)} }} {{ {round(gamma(number + alpha + 1.0), 3)} }}")

            i = sm.integrate(f)
            st.latex(rf"=\frac{{ {x}^{{{number + 1}}} }}{{ {number + 1} }}")
            st.markdown("##### Plotting of Reimann-Liouvelle integral: #####")

            x = np.linspace(0, 1, 100)
            y = x ** (number + 1) / (number + 1)

            plt.plot(x, y)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"y = x^({number + 1}) / ({number + 1})")

            fig = plt.gcf()
            st.pyplot(fig)

            break

def constant_multi_var(f, alpha):
    x = sm.Symbol('x')
    a = sm.Symbol('a')
    t = sm.Symbol('t')
    s = sm.Symbol('s')
    n = sm.Symbol('n')
    α = sm.Symbol('α')
    numbers = [int(i) for i in range(1, 1000)]
    coeff = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]

    match = re.search(r'(\d+)\*x\^(\d+)', f)
    if match:
        c = int(match.group(1))
        number = int(match.group(2))

        st.markdown(
            f"##### Using Reimann-Liovelle integral from $'a'$ to $'x'$ to the order $α$, we get, #####")
        st.latex(
            rf"_{{{a}}}I_{{{x}}}^ α(f) =c × \frac{{ {x}^{{{n + α}}} }}{{ {{\Gamma({α})}} }}×\frac {{\Gamma({α})×\Gamma({n + 1})}} {{\Gamma({α + n + 1}) }}")
        st.markdown(f"##### Here, ${α}={alpha}$ and when ${a}={0}$,")
        st.latex(
            rf"_{{{0}}}I_{{{x}}}^ {{{alpha}}}({f}) = {c} × \frac{{{x}^{{{round(alpha - 1 + number + 1, 2)}}}}} {{\Gamma({alpha})}} ×\frac{{ \Gamma{round(alpha - 1 + 1, 2)}×\Gamma{number + 1} }}{{ \Gamma{number + alpha + 1} }}")
        st.latex(
            rf"={c} × \frac{{ {x}^{{{number + alpha}}}×{gamma(number + 1.0)} }} {{ {round(gamma(number + alpha + 1.0), 3)} }}")
        st.latex(
            rf"= \frac{{ {x}^{{{number + alpha}}}×{c * (gamma(number + 1.0))} }} {{ {round(gamma(number + alpha + 1.0), 3)} }}")

        st.markdown("##### Plotting of Reimann-Liouvelle integral: #####")
        plt.style.use(['tableau-colorblind10', 'ggplot'])

        x = np.linspace(0, 1, 100)

        y = (x ** (number + alpha) * gamma(number + 1.0)) / round(gamma(number + alpha + 1.0), 3)

        plt.plot(x, y)
        plt.xlabel("g(x)")
        plt.ylabel("f(x)")
        plt.title(
            rf"g(x) = x^({number + alpha})×{c * (gamma(number + 1.0))} / {round(gamma(number + alpha + 1.0), 3)} ")
        fig = plt.gcf()
        st.pyplot(fig)

def const_multi_var_normal(f, alpha):
    x = sm.Symbol('x')
    a = sm.Symbol('a')
    t = sm.Symbol('t')
    s = sm.Symbol('s')
    n = sm.Symbol('n')
    α = sm.Symbol('α')

    match = re.search(r'(\d+)\*x\^(\d+)', f)
    if match:
        c = int(match.group(1))
        number = int(match.group(2))

        st.markdown(
            f"##### Using Reimann-Liouvelle integral from $'a'$ to $'x'$ to the order $α$, we get, #####")
        st.latex(
            rf"_{{{a}}}I_{{{x}}}^ α(f) =c × \frac{{ {x}^{{{n + α}}} }}{{ {{\Gamma({α})}} }}×\frac {{\Gamma({α})×\Gamma({n + 1})}} {{\Gamma({α + n + 1}) }}")
        st.markdown(f"##### Here, ${α}={alpha}$ and when ${a}={0}$,")
        st.latex(
            rf"_{{{0}}}I_{{{x}}}^ {{{alpha}}}({f}) = {c} × \frac{{{x}^{{{round(alpha - 1 + number + 1, 2)}}}}} {{\Gamma({alpha})}} ×\frac{{ \Gamma{round(alpha - 1 + 1, 2)}×\Gamma{number + 1} }}{{ \Gamma{number + alpha + 1} }}")
        st.latex(
            rf"={c} × \frac{{ {x}^{{{number + alpha}}}×{gamma(number + 1.0)} }} {{ {round(gamma(number + alpha + 1.0), 3)} }}")
        st.latex(
            rf"= \frac{{ {x}^{{{number + alpha}}}×{c * (gamma(number + 1.0))} }} {{ {round(gamma(number + alpha + 1.0), 3)} }}")

        st.markdown("##### Plotting of Reimann-Liouvelle integral: #####")
        plt.style.use(['tableau-colorblind10', 'ggplot'])

        x = np.linspace(0, 1, 100)

        y = (x ** (number + alpha) * gamma(number + 1.0)) / round(gamma(number + alpha + 1.0), 3)

        plt.plot(x, y)
        plt.xlabel("g(x)")
        plt.ylabel("f(x)")
        plt.title(
            rf"g(x) = x^({number + alpha})×{c * (gamma(number + 1.0))} / {round(gamma(number + alpha + 1.0), 3)} ")
        fig = plt.gcf()
        st.pyplot(fig)

def sine_frac(f, alpha):
    x = sm.Symbol('x')
    a = sm.Symbol('a')
    α = sm.Symbol('α')
    π = sm.Symbol('π')
    inf = sm.Symbol('∞')

    decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]

    if f == "sin(x)" and alpha in decimal:
        st.markdown(
            f"##### Using Reimann-Liovelle integral from $'a'$ to $'x'$ to the order $α$, we get, #####")
        st.latex(
            rf"_{{{a}}}I_{{{x}}}^ α[ {f} ] = sin({x}-\frac{π}{2} {α})")
        st.markdown(f"##### Here, ${α}={alpha}$ and taking the base point ${a}={-inf}$,")
        st.latex(rf"_{{{-inf}}}I_{{{x}}}^ α[ {f} ] = sin({x}-\frac{π}{2}×{alpha})")
        st.latex(rf"= sin({x}-{0.5 * alpha}{π})")

        x_min = -np.pi
        x_max = np.pi

        st.markdown("##### Plotting of Reimann-Liouvelle integral: #####")
        plt.style.use(['tableau-colorblind10', 'ggplot'])

        x = sm.symbols('x')
        y = sm.sin(x - 0.5 * alpha * sm.pi)

        x_vals = np.linspace(x_min, x_max, 1000)
        y_vals = [y.subs(x, x_val) for x_val in x_vals]

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals)
        ax.set_xlabel("x (radians)")
        ax.set_ylabel("y")
        ax.set_title(f"Plot of sin({x} - {0.5 * alpha}*{π})")

        # Show the plot using Streamlit
        st.pyplot(fig)
    elif f == "sin(x)" and alpha == 1:
        st.markdown(
            f"##### Using Reimann-Liovelle integral from $'a'$ to $'x'$ to the order $α$, we get, #####")
        st.latex(
            rf"_{{{a}}}I_{{{x}}}^ α[ {f} ] = sin({x}-\frac{π}{2} {α})")
        st.markdown(f"##### Here, ${α}={alpha}$ and taking the base point ${a}={-inf}$,")
        st.latex(rf"_{{{-inf}}}I_{{{x}}}^ α[ {f} ] = sin({x}-\frac{π}{2}×{alpha})")
        st.latex(rf"= sin({x}-{0.5 * alpha}{π})")
        st.latex(rf"= -sin({0.5 * alpha}{π}-{x})")
        st.latex(rf"= -cos({x})")

        x_min = -np.pi
        x_max = np.pi

        st.markdown("##### Plotting of Reimann-Liouvelle integral: #####")
        plt.style.use(['tableau-colorblind10', 'ggplot'])

        x = sm.symbols('x')
        y = sm.sin(x - 0.5 * alpha * sm.pi)

        x_vals = np.linspace(x_min, x_max, 1000)
        y_vals = [y.subs(x, x_val) for x_val in x_vals]

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals)
        ax.set_xlabel("x (radians)")
        ax.set_ylabel("y")
        ax.set_title(f"Plot of sin({x} - {0.5 * alpha}*{π})")
        st.pyplot(fig)

def reimann_liovelle_exponent(f, alpha):
    decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    numbers = (int(i) for i in range(0, 1000))
    k = sm.Symbol('k')
    x = sm.Symbol('x')
    alp = sm.Symbol('α')
    a = sm.Symbol('a')
    e = sm.Symbol('e')
    for number in numbers:
        if f == f"e^({number}{x})" or f == f"e^({number}*{x})" and alpha in decimal:
            st.markdown(
                f"##### Using Reimann-Liouvelle integral from $'a'$ to $'x'$ to the order $α$, we get, #####")
            st.latex(rf"_{{{0}}}I_{{{x}}}^ α(f) =\frac {{{{ {e}^{{{k}{x}}} }}}} {{{{ {k}^{alp} }}}} ")
            st.markdown(f"##### Here, ${alp}={alpha}$, so,")
            st.latex(rf"_{{{0}}}I_{{{x}}}^ α(f) =\frac {{{{ {e}^{{{number}{x}}} }}}} {{{{ {number}^{{{alpha}}} }}}} ")
            st.latex(rf"=\frac {{{{ {e}^{{{number}{x}}} }}}} {{{{ {round(pow(number, alpha), 3)} }}}} ")
            st.markdown("##### Plotting of Reimann-Liouvelle integral: #####")

            f = sm.exp(number * x) / number ** alpha

            # plot the function
            x_values = np.linspace(0, 1, 100)
            y_values = [sm.N(f.subs(x, xval)) for xval in x_values]

            plt.plot(x_values, y_values)
            plt.xlabel("g(x)")
            plt.ylabel("f(x)")
            plt.title(fr"$g(x) = \frac{{{e}^{{{number}{x}}}}}{{{round(number ** alpha, 3)}}}$")

            fig = plt.gcf()
            st.pyplot(fig)
