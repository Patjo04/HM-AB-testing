# Written by H&M 2025-01-31
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib.axes import Axes



def plot_cdf(data: np.ndarray, label: str, ax: Axes, color: str = "blue", linewidth: float = 3):
    sorted_data = np.sort(data)
    position = st.rankdata(sorted_data, method='ordinal')
    cdf = position / data.shape[0]

    sorted_data = np.hstack((sorted_data, 1))
    cdf = np.hstack((cdf, 1))

    return ax.plot(sorted_data, cdf, color=color, linestyle='solid', label=label, linewidth=linewidth)

def plot_summary(
    dict2plot: dict,
    mark: dict = None 
):
  """
  :dict2plot - dict with the structure
    - key: stat test name as str
    - value: (p_values_ab, p_values_aa, reqSS, color), where
      - p_values_aa, p_values_ab - np.arrays of floats
      - reqSS - int - required sample size 
      - color: str
  :mark - dict with the structure
    - label: one of the keys from dict2plot to mark
    - color: color for marking
  :return: fig
  """
  if mark!=None:
    dict2plot[mark['label']][-1]=mark['color']
    
  fig = plt.figure(constrained_layout=False, figsize=(5 * 3, 3 * 3), dpi=100)
  gs = fig.add_gridspec(3, 4)
  ax_fpr = fig.add_subplot(gs[:2, :2])
  ax_tpr = fig.add_subplot(gs[:2, 2:4])
  ax_powers = fig.add_subplot(gs[2, 2:4])
  ax_obs_num = fig.add_subplot(gs[2,:2])

  fig.subplots_adjust(left=0.2, wspace=1., hspace=0.4)

  _ = ax_fpr.set_title("FPR, CDF for p-values under H0")
  _ = ax_tpr.set_title("TPR (Sensitivity, Recall), CDF for p-values under H1")
  _ = ax_powers.set_title("Test Power (TPR at 0.05)")
  _ = ax_obs_num.set_title("Average Required Sample Size")

  for title, (ab_pvals, aa_pvals, _, color) in dict2plot.items():
        plot_cdf(ab_pvals, title, ax_tpr, color, linewidth=2)
        plot_cdf(aa_pvals, title, ax_fpr, color, linewidth=2)

  _ = sns.lineplot(x=np.arange(0,1,0.01),y=np.arange(0,1,0.01), color = "grey", ax = ax_fpr)
  _ = ax_fpr.grid(True)
  

  _ = sns.lineplot(x=np.arange(0,1,0.01),y=np.arange(0,1,0.01), color = "grey", ax = ax_tpr)
  _ = ax_tpr.axvline(0.05, color='grey')
  _ = ax_fpr.axvline(0.05, color='grey')
  _ = ax_tpr.grid(True)

  tests_powers = []
  tests_labels = []
  tests_colours = []
  tests_reqSS = []

  for title, (ab_pvals, _, reqSS, color) in dict2plot.items():
      tests_labels.append(title)
      tests_colours.append(color)
      tests_powers.append(np.mean(ab_pvals < 0.05))
      tests_reqSS.append(reqSS)
  ax_powers.barh(np.array(tests_labels), np.array(tests_powers), color=np.array(tests_colours))
  ax_obs_num.barh(np.array(tests_labels), np.array(tests_reqSS), color=np.array(tests_colours))
  
  return fig
