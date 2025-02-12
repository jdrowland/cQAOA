import marimo

__generated_with = "0.10.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import os
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    return json, mo, np, os, pd, plt, sns


@app.cell
def _(mo, os):
    data_dir = "../data"
    all_files = os.listdir(data_dir)
    output_files = [s for s in all_files if "output" in s and ".json" in s]
    dropdown = mo.ui.dropdown(output_files, label="File to plot:")
    mo.vstack([dropdown])

    return all_files, data_dir, dropdown, output_files


@app.cell
def _(data_dir, dropdown, json):
    data_file = data_dir + "/" + dropdown.value
    with open(data_file, "r", encoding="utf8") as f:
        data_dict = json.load(f)
    return data_dict, data_file, f


@app.cell
def _(data_dict):
    print(data_dict.keys())
    return


@app.cell
def _(data_dict, mo):
    p = data_dict["input"]["p"]
    rounds = data_dict["input"]["rounds"]
    regular_energy = data_dict["regular_qaoa"]["energy"]
    instance = data_dict["input"]["key"]
    input_file = data_dict["input"]["hdf_file"]
    mo.md(f'''
    Problem instance is {instance} in file {input_file}.\n
    Optimizing with p= {p} and {rounds} rounds.\n
    Regular QAOA final energy = {regular_energy}.
    ''')
    return input_file, instance, p, regular_energy, rounds


@app.cell
def _(data_dict, np, plt, regular_energy):
    cyclic_energies = np.array(data_dict["cyclic_qaoa"]["energies"])
    fig, ax = plt.subplots()
    ax.plot(cyclic_energies, '.', label="Cyclic energies")
    ax.hlines([regular_energy], 0.0, float(cyclic_energies.size - 1), 'k',\
              label="QAOA energy")
    ax.set_xticks(range(cyclic_energies.size))
    ax.set_xlabel("Round")
    ax.set_ylabel("Energy")
    plt.legend()
    return ax, cyclic_energies, fig


@app.cell
def _(data_dict, np, plt):
    refs = np.array(data_dict["cyclic_qaoa"]["references"])
    fig2, ax2 = plt.subplots()
    ax2.grid()
    ax2.imshow(refs, extent=[0, refs.shape[1], 0, refs.shape[0]])
    ax2.set_xlabel("Spin")
    ax2.set_ylabel("Round")
    ax2.set_title("Reference states")
    return ax2, fig2, refs


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
