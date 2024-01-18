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
        return
    elif (f in numbers) and (alpha in decimal):
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


def caputo_x_raised_num(f, alpha):
    x = sm.Symbol('x')
    a = sm.Symbol('a')
    t = sm.Symbol('t')
    s = sm.Symbol('s')
    n = sm.Symbol('n')
    α = sm.Symbol('α')
    L = sm.Symbol('L')
    beta = sm.Symbol('β')
    numbers = (int(i) for i in range(1, 1000))
    decimal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


    for number in numbers:
        if f == f"x^{number}" and alpha in decimal:
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

        elif f == f"x^{number}" and alpha == 1:
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
            st.latex(f"= {(gamma(1.0 + number)) / round(gamma(1.0 + number - alpha), 2)}{x}^{{{number - alpha}}}")
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


def caputo_wolfram(f, alpha):
    L = sm.Symbol('L')
    x = sm.Symbol('x')
    a = sm.Symbol('a')
    t = sm.Symbol('t')
    s = sm.Symbol('s')
    n = sm.Symbol('n', constant=True)
    alp = sm.Symbol('α')
    beta = sm.Symbol('β')
    numbers = (int(i) for i in range(1, 1000))
    decimal = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ceil = ceiling(alpha)
    decimal_numbers = [float(x) for x in range(1001)]


    if (f) and (alpha in decimal):
        st.markdown(f"##### Using Caputo fractional derivative from $'a'$ to $'x'$ to the order ${alp}$, we get, #####")
        st.latex(rf"^c_{{{a}}}D_{{{x}}}^ α(f) = {{_{{{a}}}I_{{{x}}}^ {{n-α}}}}  [ \frac{{d^n}} {{dx^n}}[f({x})] ] ")
        st.markdown(
            f"##### Here, ${alp}={alpha}$, #####\n##### $n=⌈{alp}⌉ = ⌈{alpha}⌉ = {ceil}$, #####\n ##### also, when ${a}={0}$, we get, #####")

        if ceil - alpha == 0:
            st.latex(
                rf"^c_{{{0}}}D_{{{x}}}^ {{{alpha}}}[{f}] = {{_{{{0}}}I_{{{x}}}^ {{{round(ceil - alpha, 1)}}}}}  [ \frac{{d^{int(ceil)}}} {{dx^{int(ceil)}}}[{f}] ] ")

            differen =  sm.diff(f)
            st.latex(rf"= {{_{{{0}}}I_{{{x}}}^ {{{round(ceil - alpha, 1)}}}}}  [{sm.latex(differen)}] ")

            st.latex(f"= {sm.latex(differen)}")

            differen_numeric = sm.lambdify(x, differen, 'numpy')

            # Generate x values
            x_values = np.linspace(-10, 10, 100)
            y_values = differen_numeric(x_values)

            fig, ax = plt.subplots()
            ax.plot(x_values, y_values)
            ax.set_xlabel('x')
            ax.set_ylabel("f '(x)")
            ax.set_title(f"Plot of {differen}")

            st.pyplot(fig)

            return


        st.latex(
            rf"^cD_{{{x}}}^ {{{alpha}}}[{f}] = {{_{{{0}}}I_{{{x}}}^ {{{round(ceil - alpha, 1)}}}}}  [ \frac{{d^{int(ceil)}}} {{dx^{int(ceil)}}}[{f}] ] ")

        differen = sm.diff(f,x)
        st.latex(rf"= {{_{{{0}}}I_{{{x}}}^ {{{round(ceil - alpha, 1)}}}}}  [{sm.latex(differen)}] ")

        st.markdown("##### Now, using Reimann-Liouville fractional integral, we get, #####")
        mathjax_code = differen.subs('x', 't')

        st.latex(
            rf"= \frac {{1}} {{\Gamma({round(ceil - alpha, 1)})}} \int_{0}^{x} ({x}-{t})^{{({round(ceil - alpha - 1, 1)})}}[{sm.latex(mathjax_code)}] dt")


        fn = mathjax_code.subs('t','x')
        gn = f"({x})^{{{round(ceil - alpha - 1, 1)}}}"

        st.markdown(f"##### Now, we know, #####")

        st.latex(f"\mathcal{{L}}^{{{-1}}}[f(s)*g(s)] = \int_{0}^{t} f(u).g(t-u) du")

        st.markdown("##### Thus, taking Laplace on both the sides, we get, #####")

        st.latex(rf"\mathcal{{L}}[^cD_{{{x}}}^ {{{alpha}}}[{f}]] = \frac {{1}} {{\Gamma({round(ceil - alpha, 1)})}} \mathcal{{L}}\left[\int_{0}^{x} ({x}-{t})^{{({round(ceil - alpha - 1, 1)})}}[{sm.latex(mathjax_code)}] dt \right]")
        st.latex(rf"\mathcal{{L}}[^cD_{{{x}}}^ {{{alpha}}}[{f}]] = \frac {{1}} {{\Gamma({round(ceil - alpha, 1)})}}[{sm.latex(fn)}*{gn}] ")
        lap_first = sm.laplace_transform(fn,x,s,noconds=True)
        lap_second = rf"\frac{{\Gamma({round((ceil - alpha - 1)+1, 1)})}} {{  {s}^{{{round((ceil - alpha - 1)+1, 1)}}}  }}"

        st.markdown("##### Taking Inverse Laplace on both the sides, we get, #####")

        st.latex(rf"^cD_{{{x}}}^ {{{alpha}}}[{f}] = \frac {{1}} {{\Gamma({round(ceil - alpha, 1)})}} \mathcal{{L}} ^{{{-1}}}\left[{sm.latex(lap_first)} * {lap_second} \right]")
