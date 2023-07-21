import math  # for gamma
import streamlit as st
import sympy as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from streamlit_option_menu import option_menu
from sympy import ceiling
import wolframalpha
from sympy import symbols, latex, pretty



st.set_page_config(page_title="fraculus")
st.markdown(
    """
    <style>
    body {
        background-color: #000000; /* Set the background color to black */
        color: #ffffff; /* Set the text color to white */
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=['Home', 'R-L fractional Derivative', 'R-L fractional Integral','Caputo fractional Derivative' ],
        icons=['house', 'book', 'book', 'book']
    )

hide_streamlit_style = """
            <style>

            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


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


def caputo_constant(f, alpha):
    x = sm.Symbol('x')
    a = sm.Symbol('a')
    t = sm.Symbol('t')
    c = sm.Symbol('c')
    alp = sm.Symbol('α')
    numbers = set(str(i) for i in range(1, 1000))
    n = [1, 2, 3, 4, ..., 1000]
    decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    ceil = ceiling(alpha)

    if (f in numbers) and (alpha == 0):
        st.latex(f)
    elif (f in numbers) and (alpha in decimal):
        st.markdown(f"##### Using Caputo fractional derivative from $'a'$ to $'x'$ to the order ${alp}$, we get, #####")
        st.latex(rf"_{{{a}}}D_{{{x}}}^ α(f) = {{_{{{a}}}I_{{{x}}}^ {{n-α}}}}  [ \frac{{d^n}} {{dx^n}}[f(t)] ] ")
        st.markdown(f"##### Here, ${alp}={alpha}$, #####\n##### $n=⌈{alp}⌉ = ⌈{alpha}⌉ = {ceil}$, #####\n ##### also, when ${a}={0}$, we get, #####")

        st.latex(rf"_{{{0}}}D_{{{x}}}^ {{{alpha}}}({f}) = {{_{{{0}}}I_{{{x}}}^ {{{round(ceil-alpha,1)}}}}}  [ \frac{{d^{int(ceil)}}} {{dx^{int(ceil)}}}[{f}] ] ")
        st.latex(rf"= {0}")

        st.markdown("##### Plotting of Caputo fractional derivative: #####")

        plt.style.use(['tableau-colorblind10', 'ggplot'])

        x = np.linspace(0, 1, 100)

        y = np.zeros_like(x)

        plt.plot(x, y)
        plt.xlabel("g(x)")
        plt.ylabel("f(x)")
        plt.title(
            f"g(x) = 0")

        fig = plt.gcf()
        st.pyplot(fig)

    elif (f in numbers) and (alpha == 1):
        st.markdown(f"##### Using Caputo fractional derivative from $'a'$ to $'x'$ to the order ${alp}$, we get, #####")
        st.latex(rf"_{{{a}}}D_{{{x}}}^ α(f) = {{_{{{a}}}I_{{{x}}}^ {{n-α}}}}  [ \frac{{d^n}} {{dx^n}}[f(t)] ] ")
        st.markdown(
            f"##### Here, ${alp}={alpha}$, #####\n##### $n=⌈{alp}⌉ = ⌈{alpha}⌉ = {ceil}$, #####\n ##### also, when ${a}={0}$, we get, #####")

        st.latex(
            rf"_{{{0}}}D_{{{x}}}^ {{{alpha}}}({f}) = {{_{{{0}}}I_{{{x}}}^ {{{round(ceil - alpha, 1)}}}}}  [ \frac{{d^{int(ceil)}}} {{dx^{int(ceil)}}}[{f}] ] ")
        st.latex(rf"= {0}")

        st.markdown("##### Plotting of Caputo fractional derivative: #####")

        plt.style.use(['tableau-colorblind10', 'ggplot'])

        x = np.linspace(0, 1, 100)

        y = np.zeros_like(x)

        plt.plot(x, y)
        plt.xlabel("g(x)")
        plt.ylabel("f(x)")
        plt.title(
            f"g(x) = 0")

        fig = plt.gcf()
        st.pyplot(fig)

def caputo_x_raised_num(f,alpha):
    x = sm.Symbol('x')
    a = sm.Symbol('a')
    t = sm.Symbol('t')
    s = sm.Symbol('s')
    n = sm.Symbol('n')
    α = sm.Symbol('α')
    beta = sm.Symbol('β')
    numbers = (int(i) for i in range(1, 1000))
    decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]

    for number in numbers:
        if f == f"x^{number}" and alpha in decimal:
            st.markdown(
                f"##### Using Caputo fractional derivative from $'a'$ to $'x'$ to the order $α$, we get, #####")
            st.latex(rf"_{{{a}}}D_{{{x}}}^ α(f) =\frac {{{{\Gamma({1}+{beta})×{x}^{{{beta}-{α}}} }}}} {{{{\Gamma({1}+{beta}-{α}) }}}}")
            st.markdown(
                f"##### Here, ${α}={alpha}$ and when ${a}={0}$, we get,")

            st.latex(rf"_{{{0}}}D_{{{x}}}^ {{{alpha}}}({f}) =\frac {{{{\Gamma({1+number})×{x}^{{{number-alpha}}} }}}} {{{{\Gamma({1+number-alpha}) }}}}")
            st.latex(rf"=\frac {{{{{gamma(1.0+number)}×{x}^{{{number-alpha}}} }}}} {{{{{round(gamma(1.0+number-alpha),2) }}}}}")

            st.markdown("##### Plotting of Caputo fractional derivative: #####")

            plt.style.use(['tableau-colorblind10', 'ggplot'])

            x = np.linspace(0, 1, 100)

            y = gamma(1.0 + number) * x**(number - alpha) / round(gamma(1.0 + number - alpha), 2)

            plt.plot(x, y)
            plt.xlabel("g(x)")
            plt.ylabel("f(x)")
            plt.title(rf"g(x) = $\frac{{\Gamma({1+number}) \cdot x^{{{number-alpha}}}}}{{\Gamma({1+number-alpha})}}$")


            fig = plt.gcf()
            st.pyplot(fig)

        elif f == f"x^{number}" and alpha==1:
            st.markdown(
                f"##### Using Caputo fractional derivative from $'a'$ to $'x'$ to the order $α$, we get, #####")
            st.latex(
                rf"_{{{a}}}D_{{{x}}}^ α(f) =\frac {{{{\Gamma({1}+{beta})×{x}^{{{beta}-{α}}} }}}} {{{{\Gamma({1}+{beta}-{α}) }}}}")
            st.markdown(
                f"##### Here, ${α}={alpha}$ and when ${a}={0}$, we get,")

            st.latex(
                rf"_{{{0}}}D_{{{x}}}^ {{{alpha}}}({f}) =\frac {{{{\Gamma({1 + number})×{x}^{{{number - alpha}}} }}}} {{{{\Gamma({1 + number - alpha}) }}}}")
            st.latex(
                rf"=\frac {{{{{gamma(1.0 + number)}×{x}^{{{number - alpha}}} }}}} {{{{{round(gamma(1.0 + number - alpha), 2)}}}}}")
            st.latex(f"= {(gamma(1.0 + number))/round(gamma(1.0 + number - alpha), 2)}{x}^{{{number - alpha}}}")
            st.markdown("##### Plotting of Caputo fractional derivative: #####")

            plt.style.use(['tableau-colorblind10', 'ggplot'])

            x = np.linspace(0, 1, 100)

            y = gamma(1.0 + number) * x ** (number - alpha) / round(gamma(1.0 + number - alpha), 2)

            plt.plot(x, y)
            plt.xlabel("g(x)")
            plt.ylabel("f(x)")
            plt.title(
                rf"g(x) = $\frac{{\Gamma({1 + number}) \cdot x^{{{number - alpha}}}}}{{\Gamma({1 + number - alpha})}}$")

            fig = plt.gcf()
            st.pyplot(fig)

def caputo_wolfram(f,alpha):
    client = wolframalpha.Client('HW2K6X-TUYKU4ULKW')

    x = sm.Symbol('x')
    a = sm.Symbol('a')
    t = sm.Symbol('t')
    s = sm.Symbol('s')
    n = sm.Symbol('n', constant=True)
    alp = sm.Symbol('α')
    beta = sm.Symbol('β')
    numbers = (int(i) for i in range(1, 1000))
    decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    ceil = ceiling(alpha)
    decimal_numbers = [float(x) for x in range(1001)]

    query = f"{ceil}th order derivative of {f}"

    if query.startswith("integral of"):
        query = "integrate " + f[12:]  # Modify the query to match the expected format


    res = client.query(query)

    output = next(res.results).text

    # Extract the right-hand side of the equation
    output_parts = output.split('=')
    if len(output_parts) > 1:
        desired_output = output_parts[1].strip()
    else:
        desired_output = output.strip()

    # Render output using MathJax
    mathjax_code = fr"{desired_output}"

    if (f) and (alpha in decimal):
        st.markdown(f"##### Using Caputo fractional derivative from $'a'$ to $'x'$ to the order ${alp}$, we get, #####")
        st.latex(rf"^c_{{{a}}}D_{{{x}}}^ α(f) = {{_{{{a}}}I_{{{x}}}^ {{n-α}}}}  [ \frac{{d^n}} {{dx^n}}[f({x})] ] ")
        st.markdown(f"##### Here, ${alp}={alpha}$, #####\n##### $n=⌈{alp}⌉ = ⌈{alpha}⌉ = {ceil}$, #####\n ##### also, when ${a}={0}$, we get, #####")

        st.latex(rf"^c_{{{0}}}D_{{{x}}}^ {{{alpha}}}[{f}] = {{_{{{0}}}I_{{{x}}}^ {{{round(ceil-alpha,1)}}}}}  [ \frac{{d^{int(ceil)}}} {{dx^{int(ceil)}}}[{f}] ] ")

        st.latex(rf"= {{_{{{0}}}I_{{{x}}}^ {{{round(ceil-alpha,1)}}}}}  [{mathjax_code}] ")

        st.markdown("##### Now, using Reimann-Liouville fractional integral, we get, #####")
        mathjax_code = mathjax_code.replace('x','t')

        st.latex(rf"= \frac {{1}} {{\Gamma({round(ceil-alpha,1)})}} \int_{0}^{x} ({x}-{t})^{{({round(ceil-alpha-1,2)})}}[{mathjax_code}] dt")


        if (ceil-alpha-1) < 0:
            mathjax_code = mathjax_code.replace('x', 't')
            st.latex(rf"= \frac {{1}} {{\Gamma({round(ceil-alpha,1)})}} \int_{0}^{x} \frac {{ [{mathjax_code}] }} {{ ({x}-{t})^{{({-(round(ceil-alpha-1,2))})}} }}  dt")

            query_1 = f'''integral of ({mathjax_code})/((x-t)^({-(round(ceil-alpha-1,2))}))dt from limits 0 to x'''

            res_1 = client.query(query_1)
            output_1 = next(res_1.results).text

            output_parts_1 = output_1.split('=')

            if len(output_parts_1) > 1:
                desired_output_1 = output_parts_1[1].strip()
            else:
                desired_output_1 = output_1.strip()

            desired_output_1 = re.sub(r'for Re\(x\)>0 ∧ Im\(x\)', '', desired_output_1)

            desired_output_1 = desired_output_1.replace('x^0.1', 'x^{{0.1}}').replace('x^0.2', 'x^{{0.2}}').replace('x^0.3', 'x^{{0.3}}').replace('x^0.4', 'x^{{0.4}}').replace('x^0.5', 'x^{{0.5}}').replace('x^0.6', 'x^{{0.6}}').replace('x^0.7', 'x^{{0.7}}').replace('x^0.8', 'x^{{0.8}}').replace('x^0.9', 'x^{{0.9}}')
            desired_output_1 = desired_output_1.replace('x^1.1', 'x^{{1.1}}').replace('x^1.2', 'x^{{1.2}}').replace(
                'x^1.3', 'x^{{1.3}}').replace('x^1.4', 'x^{{1.4}}').replace('x^1.5', 'x^{{1.5}}').replace('x^1.6',
                                                                                                          'x^{{1.6}}').replace(
                'x^1.7', 'x^{{1.7}}').replace('x^1.8', 'x^{{1.8}}').replace('x^1.9', 'x^{{1.9}}')

            desired_output_1 = desired_output_1.replace('x^2.1', 'x^{{2.1}}').replace('x^2.2', 'x^{{2.2}}').replace(
                'x^2.3', 'x^{{2.3}}').replace('x^2.4', 'x^{{2.4}}').replace('x^2.5', 'x^{{2.5}}').replace('x^2.6',
                                                                                                          'x^{{2.6}}').replace(
                'x^2.7', 'x^{{2.7}}').replace('x^2.8', 'x^{{2.8}}').replace('x^2.9', 'x^{{2.9}}').replace('x^3.0',
                                                                                                          'x^{{3.0}}').replace(
                'x^3.1', 'x^{{3.1}}').replace('x^3.2', 'x^{{3.2}}').replace('x^3.3', 'x^{{3.3}}').replace('x^3.4',
                                                                                                          'x^{{3.4}}').replace(
                'x^3.5', 'x^{{3.5}}').replace('x^3.6', 'x^{{3.6}}').replace('x^3.7', 'x^{{3.7}}').replace('x^3.8',
                                                                                                          'x^{{3.8}}').replace(
                'x^3.9', 'x^{{3.9}}').replace('x^4.0', 'x^{{4.0}}').replace('x^4.1', 'x^{{4.1}}').replace('x^4.2',
                                                                                                          'x^{{4.2}}').replace(
                'x^4.3', 'x^{{4.3}}').replace('x^4.4', 'x^{{4.4}}').replace('x^4.5', 'x^{{4.5}}').replace('x^4.6',
                                                                                                          'x^{{4.6}}').replace(
                'x^4.7', 'x^{{4.7}}').replace('x^4.8', 'x^{{4.8}}').replace('x^4.9', 'x^{{4.9}}').replace('x^5.0',
                                                                                                          'x^{{5.0}}').replace(
                'x^5.1', 'x^{{5.1}}').replace('x^5.2', 'x^{{5.2}}').replace('x^5.3', 'x^{{5.3}}').replace('x^5.4',
                                                                                                          'x^{{5.4}}').replace(
                'x^5.5', 'x^{{5.5}}').replace('x^5.6', 'x^{{5.6}}').replace('x^5.7', 'x^{{5.7}}').replace('x^5.8',
                                                                                                          'x^{{5.8}}').replace(
                'x^5.9', 'x^{{5.9}}').replace('x^6.0', 'x^{{6.0}}').replace('x^6.1', 'x^{{6.1}}').replace('x^6.2',
                                                                                                          'x^{{6.2}}').replace(
                'x^6.3', 'x^{{6.3}}').replace('x^6.4', 'x^{{6.4}}').replace('x^6.5', 'x^{{6.5}}').replace('x^6.6',
                                                                                                          'x^{{6.6}}').replace(
                'x^6.7', 'x^{{6.7}}').replace('x^6.8', 'x^{{6.8}}').replace('x^6.9', 'x^{{6.9}}').replace('x^7.0',
                                                                                                          'x^{{7.0}}').replace(
                'x^7.1', 'x^{{7.1}}').replace('x^7.2', 'x^{{7.2}}').replace('x^7.3', 'x^{{7.3}}').replace('x^7.4',
                                                                                                          'x^{{7.4}}').replace(
                'x^7.5', 'x^{{7.5}}').replace('x^7.6', 'x^{{7.6}}').replace('x^7.7', 'x^{{7.7}}').replace('x^7.8',
                                                                                                          'x^{{7.8}}').replace(
                'x^7.9', 'x^{{7.9}}').replace('x^8.0', 'x^{{8.0}}').replace('x^8.1', 'x^{{8.1}}').replace('x^8.2',
                                                                                                          'x^{{8.2}}').replace(
                'x^8.3', 'x^{{8.3}}').replace('x^8.4', 'x^{{8.4}}').replace('x^8.5', 'x^{{8.5}}').replace('x^8.6',
                                                                                                          'x^{{8.6}}').replace(
                'x^8.7', 'x^{{8.7}}').replace('x^8.8', 'x^{{8.8}}').replace('x^8.9', 'x^{{8.9}}').replace('x^9.0',
                                                                                                          'x^{{9.0}}').replace(
                'x^9.1', 'x^{{9.1}}').replace('x^9.2', 'x^{{9.2}}').replace('x^9.3', 'x^{{9.3}}').replace('x^9.4',
                                                                                                          'x^{{9.4}}').replace(
                'x^9.5', 'x^{{9.5}}').replace('x^9.6', 'x^{{9.6}}').replace('x^9.7', 'x^{{9.7}}').replace('x^9.8',
                                                                                                          'x^{{9.8}}').replace(
                'x^9.9', 'x^{{9.9}}')

            desired_output_1 = desired_output_1.replace('sqrt', '√')

            desired_output_1 = re.sub(r'x\^(\d+)/(\d+)', r'\\frac{x^{\1}}{\2}', desired_output_1)

            mathjax_code_1 = desired_output_1

            st.latex(rf"= \frac {{1}} {{\Gamma({round(ceil-alpha,1)})}}.[{mathjax_code_1}]")

            if f=="sin(x)":
                st.markdown(rf"$_m F _n(a_1,...,a_m; b_1,...,b_n;z)$ <i>is the generalized Gauss's hypergeometric function</i>",unsafe_allow_html=True)

            st.latex(rf"= \frac {{[{mathjax_code_1}] }}  {{ {round(math.gamma((ceil-alpha)),3)} }}")


def main():

    if selected == 'Home':
        st.image("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDABQODxIPDRQSEBIXFRQYHjIhHhwcHj0sLiQySUBMS0dARkVQWnNiUFVtVkVGZIhlbXd7gYKBTmCNl4x9lnN+gXz/2wBDARUXFx4aHjshITt8U0ZTfHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHz/wAARCAGPA70DASIAAhEBAxEB/8QAGwABAAMBAQEBAAAAAAAAAAAAAAQFBgMCAQf/xABPEAABAwIDAwYICwYFAwQCAwABAAIDBBEFEiETMUEGFDJRcbEiNGFygZGhwRUjMzVCUlRzktHhFlOCk9LwJENilPElg7JjZKKjwuJEVaT/xAAXAQEBAQEAAAAAAAAAAAAAAAAAAQID/8QAGREBAQEBAQEAAAAAAAAAAAAAAAERMQIS/9oADAMBAAIRAxEAPwDJoiLSCIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIu1JTSVc7Yoxqd56h1r3iNKKOsfCCS0WIJ4iygjIiKgiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIg0PJ51MI3MjzGci7yRw6go3KB0MskckUjHuF2PAOot/ZXXk8BFT1VQ7cNPULlURJc4uO8m5UUREVQREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQX0X+G5NPduMl/abdyoVfY1/h8KpafcdL+gfmVQqAiIqCIulNE2adkb5GxNcdXu3BBzRWEcFJUUlSYmyMkgbnDnOuHC9tRbRcqKnp6n4p8rmTvuI/q34X7dygiIpFc5vODGyNsbYvixYWLrHefKo6AiIqCIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiKRR0UlYJdlYuY3NbrUEdEILSQ4EEaEHgioIiICIiAiIgIiICL61pc4NG8mwVpi+Fc1+OgBMX0h9U/koKpdqOPbVcMfBzwD2XXFWOAx7TEmHgxpd7veg78pJM1XFHwYy/rP6KnU3F5Nric5HA5R6BZRZYZIHBsrHMcRezhY2QeF9a0ucGtBJO4AL4plDUOjY+PnXNmby5rLvd5AR+aDy7D5Yoy+oLINLhr3Wc7sG9RVNNVTREmCn2j/3lQcx/Du9d1CJuSTxQWNNSzyYc4U8TnunfYkbg1vWeFyfYuYZDRVUuaYPfELxlmoc63X5D3KMaiYwiEyO2TdzL6epc0EmqrDVAF0bGv0zvG95AsCVGREBERUEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBXvJlvjDz/pHeqJfRI9os17gL3sDpdQXnKGOlBDrltSRewbo4eVUSvsVArsJhrGjwmdL06H2qhQERFQREQEREBERB3oXRtrIXSuDWNcHEnyarUS10b8OkqYWbZgBBadL8Csgr3k88SwVNK/okXt5CLH3KKo3kOe4tblBNw297K65NsANRM7QNAF/afcqWRhjkcx29pIKvaH/AA/J6eXcX5rH/wCKIiQPjxAS0zomMmfd8T7al28gnyrxNLHFXRc5jMgiijaWHdewvf2qPQxCSYPdK2JkdnucTY2B4dZXOomM9TJM4avcXEIPdbBzarkiBu0HwT1tOoPqXBdqiqlqA0PIDWABoAtYDd/ZXFAREVBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQXmASNngnopNQRcdh0PuVNNG6GZ8b+kxxBXXD6jmtbFLezQbO7DvU7lDT7OqbO0eDKNe0fpZQVKIioIiICIiAiIgKwwOXZYlGL2DwWn++0KvXqKQxSskbvY4OHoUEzGodjiUvU/wx6f1urDEv8PgNPDuL8tx6Ln2r7jtOJ56SRmokOS/adO8rlylk+NgiG5rS63bp7kVSoiKoIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIusL44w8ujzyaZL7h13HFdMRaxtW4RsDAAA4DcHWGYD03UEZX/wA5YB1yw94/RUCtuTtRs6p8DujKNO0fpdBUopGIU/NayWK2gN29h3KOgIiKgiIgIiICIiDVUIFXQUTjrsyD6WghUmOSbTE5BwYA0epWnJyTNSPYfoP9h/sqgqZNtUyyfXeT7VFc0RFUEREBERAREQEREBERAREQEREBALmwReo5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0kzHMEknSOzbr7EEmkc5+E4jmJd8m7XXXNvXyp1wOjI3CR4d2qNHWzxRSRMLAyTV42bdfYvdLUR83kpagkRPOdrgL5HDjbqQfMPE4qI3x5msa7OXHRoDd5Pr9q74vK+KplpYiWQZs2Vp0ffW561FfUTNgNIJs0AdezdxPX1+tfJqqadjWyPzBoAGgBIG654+lBLwPx8k9ERPLuyyrlNbNHSUsjIX7SaZuV7gDZjeodZKhICIioIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiIJb8Rlc+OTLGJYwBtADmcB162XKqqTUymR0bGFxLjkG8nfv1XFFAXaWo2lLBDltss2t99yuKIClR1zm0zad0MUkTXZrOB39dwQoqIOtRUSVLw6QjwRla0CwaOoBckRAREVBWvJ3wa6R50yQuPcqpWWEu2UFfL9WAtHaTZQcMJGbFKb7wFfMSOfE6kjjKR7V2wJoOKwk7m3cT1WBUZl6iub1yS95QTeUZHwllH0I2hVanY1JtcVqHX0Dsvq0UFAREVHeCrdDBJCWMkjkIJa++hHEWIXp9dJJTNheyNwZfK4jVt94HD2KMigmOxOodLC9oYzYkFrWtsCQLXPWbKPNLtn5tmyMfVYNPzXNEHqJm0lYy9szgFcOa+fHoqaK7Yad4DQNzQN57SqXcbhSTiFUZmzbS0jTfMGgXPl019KBO+WprZpYg8l7z0b7ibW9ysMQmlpqemfG7LK+PZySMdqC36N+HlVcyvqY3PdHLkL25TlAGnUOr0Ly2qmZAYWv+LuTawuCd9jvCDiiIqCIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiIPUYaZGiQkMJGYjgFNqpaaCmdS0UrpWyPzPkLcug3NUBFBZCWko4HupZnSzyxhhBbbZ36WvFeaJ9HSxx1Rlc6pZciHLoTwN1Xogn4YWyVMrp2NewMe+QuFza3C+7W2q+VGU4VC8xxte6V2UtaAcoA0J469aiRzPia9rHWEjcrtN4XrnMpgEGf4sXsLC+vC++3kQckRFQREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREH//Z")
        st.markdown(
            """
            <style>
            .styled-box {
                border: 2px solid #ffffff; 
                background-color: #1f1f1f; 
                padding: 20px; 
                border-radius: 10px; 
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.25); 
                text-align: center;
            }
            .styled-box h2 {
                color: #ffffff; /* Set the text color to white */
                font-size: 28px; /* Increase the font size for the heading */
            }
            .styled-box p {
                color: #ffffff; /* Set the text color to white */
                font-size: 18px; /* Increase the font size for the paragraph */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="styled-box">
                <h1>Welcome to Fraculus!</h1>
                <h5><i>"Explore the World of Fractional Calculus"</i></h5>
            </div>
            <br>
            """,
            unsafe_allow_html=True
        )
        with st.container():
            st.header("▸What is Fractional Calculus?")
            st.markdown(
                """
                <div style="border: 2px solid #FFFFFF; padding: 20px; border-radius: 10px;">
                <p style="font-size: 20px; color: #FFFFFF;">
                Fractional calculus is a branch of mathematics that generalizes the concepts of differentiation and integration to non-integer orders.</p> 
                <p style="font-size: 20px; color: #FFFFFF;"> It deals with fractional derivatives and integrals, which allow the analysis and modeling of phenomena exhibiting fractional behavior.
                </p>
                <img src="data:image/jpeg;base64,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", width=675>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>",unsafe_allow_html=True)

        st.header("▸The Logic")
        with st.container():
            st.markdown(
                f"""
                <div style="border: 2px solid #FFFFFF; padding: 20px; border-radius: 10px;">
                <p style="font-size: 20px; color: #FFFFFF;">
                Consider the upward parabolic equation: </p>
                <img src="data:image/jpeg;base64,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", width=655>
                
                <p style="font-size: 20px; color: #FFFFFF;">
                Now, on differentiating the above Equation to it’s first order, we get,</p>
                <img src="data:image/jpeg;base64,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", width=680>
                <p style="font-size: 20px; color: #FFFFFF;">
                On further differentiating the above equation, we get,
                <p>
                <img src="data:image/jpeg;base64,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", width=680>
                </div>
                <br>
                """,
                unsafe_allow_html=True
            )
            st.info("#### Now, let us focus upon the transition of curves in the graph! ####")
            st.markdown(
                f"""
                <div style="border: 2px solid #FFFFFF; padding: 20px; border-radius: 10px;">
                <img src="https://s12.gifyu.com/images/SuXCi.gif" width="670">
                
                <p style="font-size: 20px; color: #FFFFFF;">
                Using differential rules, the parabolic curve gradually transforms into a line passing through origin, which further transforms into a line parallel to x-axis
                </p>
                <br>
                <p style="font-size: 20px; color: #FFFFFF;">
                Now, fractional calculus gives a visualization about the motion/transition of curve when it switches from one phase to another.
                </p>
                <br>
                <p style="font-size: 20px; color: #FFFFFF;">
                It gives an idea about what actually happens during this  transitional phase, in other words, it tells  us  what  happens when (𝑛)𝑡ℎ order derivative gets converted to (𝑛+1)𝑡ℎ order derivative. 
                </p>
                </div>
                <br>
                """, unsafe_allow_html=True)


    if selected == 'R-L fractional Integral':
        st.markdown("<h1 style='text-align: center;'>🎓 Reimann-Liouvelle Integral 🎓</h1>", unsafe_allow_html=True)
        st.markdown("___")
        numbers = list(str(i) for i in range(1, 1000))
        coeff = list(int(i) for i in range(1, 1000))
        x = sm.Symbol('x')
        decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        st.markdown("### ⁍ Enter any function: ###")
        f = st.text_input("")
        st.markdown("### ⁍ Enter the value of α (alpha):  ###")

        alpha = st.number_input("", 0.0, 1.0, step=0.01, format="%.2f")

        decpow = ["{:.1f}".format(num) for num in np.arange(1.0, 1000.1, 0.1)]

        fraccoef = ["{:.1f}".format(num) for num in np.arange(1.0, 100.1, 0.1)]
        fracpow = ["{:.1f}".format(num) for num in np.arange(1.0, 100.1, 0.1)]

        if f in numbers:
            if alpha in decimal:
                reimann_liovelle_for_constant_dec(f, alpha)
            elif alpha == 1:
                normal_integration(f, alpha)
            elif alpha == 0:
                st.latex(f)

        for fp in fracpow:
            for fc in fraccoef:
                if f == f"{fc}*x^{fp}" and alpha in decimal:
                    floating_coeff_x_pow(f, alpha)
                if f == f"{fc}*x^{fp}" and alpha == 0:
                    st.latex(f"{{{fc}}}×x^{{{fp}}}")

        for dec in decpow:
            if f == f"x^{dec}" and alpha in decimal:
                floating_x_power(f, alpha)
            if f == f'x^{dec}' and alpha == 0:
                st.latex(f'{{x^{{{dec}}}}}')

        for number in numbers:
            if f == f"x^{number}" and alpha in decimal:
                reimann_liovelle_for_variable_dec(f, alpha)
                break
            elif f == f"x^{number}" and alpha == 1:
                normal_integration_var(f, alpha)
                break
            elif f == f"x^{number}" and alpha == 0:
                st.latex(f'{{x^{{{number}}}}}')
                break

        for number in numbers:
            for c in coeff:
                if f == f"{c}*x^{number}" and alpha in decimal:
                    constant_multi_var(f, alpha)
                    break
                elif f == f"{c}*x^{number}" and alpha == 1:
                    const_multi_var_normal(f, alpha)
                    break
                elif f == f"{c}*x^{number}" and alpha == 0:
                    st.latex(f)
                    break

        if f == f"sin({x})" and alpha == 0:
            st.latex(f)
        if f == f"sin({x})" and alpha in decimal or alpha == 1:
            sine_frac(f, alpha)

        for number in numbers:
            if f == f"e^({number}{x})" or f == f"e^({number}*{x})" and alpha in decimal:
                reimann_liovelle_exponent(f, alpha)


    if selected == 'Caputo fractional Derivative':
        st.markdown("<h1 style='text-align: center;'>🎓 Caputo Fractional Derivative 🎓</h1>", unsafe_allow_html=True)
        st.markdown("___")
        numbers = list(str(i) for i in range(1, 1000))
        coeff = list(int(i) for i in range(1, 1000))
        k = list(int(i) for i in range(1, 1000))
        x = sm.Symbol('x')
        decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        st.markdown("### ⁍ Enter any function: ###")
        f = st.text_input("")
        st.markdown("### ⁍ Enter the value of α (alpha):  ###")

        alpha = st.number_input("", 0.0, 1.0, step=0.01, format="%.2f")

        decpow = ["{:.1f}".format(num) for num in np.arange(1.0, 1000.1, 0.1)]

        fraccoef = ["{:.1f}".format(num) for num in np.arange(1.0, 100.1, 0.1)]
        fracpow = ["{:.1f}".format(num) for num in np.arange(1.0, 100.1, 0.1)]

        if f in numbers:
            caputo_constant(f,alpha)

        for number in numbers:
            if f == f"x^{number}" and (alpha in decimal or alpha==1):
                caputo_x_raised_num(f,alpha)
                break
            elif f == f"x^{number}" and alpha == 0:
                st.latex(f'{{x^{{{number}}}}}')
                break

        if f and (alpha in decimal):
            caputo_wolfram(f,alpha)

if __name__ == '__main__':
    main()


