"""
Functions that plot the results from the simulated experiments.

@author: Noah Burrell <burrelln@umich.edu>
"""

import seaborn as sns
import matplotlib 
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
import pandas as pd

"""
Global dictionaries used to store LaTeX formatting for labels used in plots.
"""

mechanism_name_map = {
        
            #Non-Parametric Mechanisms
            "BASELINE: MSE": r'MSE',
            "DMI: 4": r'DMI',
            "OA: 0": r'OA',
            "Phi-DIV: CHI_SQUARED": r'$\Phi$-Div: $\chi^2$',
            "Phi-DIV: KL": r'$\Phi$-Div: KL',
            "Phi-DIV: SQUARED_HELLINGER": r'$\Phi$-Div: $H^2$',
            "Phi-DIV: TVD": r'$\Phi$-Div: TVD',
            "PTS: 0": r'PTS',
            #Parametric Mechanisms
            "MSE_P: 0": r'MSE$_P$',
            "Phi-DIV_P: CHI_SQUARED": r'$\Phi$-Div$_P$: $\chi^2$',
            "Phi-DIV_P: KL": r'$\Phi$-Div$_P$: KL',
            "Phi-DIV_P: SQUARED_HELLINGER": r'$\Phi$-Div$_P$: $H^2$', 
            "Phi-DIV_P: TVD": r'$\Phi$-Div$_P$: TVD',
            
    }

mechanism_color_map = {
        
            #Non-Parametric Mechanisms
            r'MSE': '#a6cee3',
            r'DMI': '#d9ef8b', 
            r'OA': '#b15928',
            r'$\Phi$-Div: $\chi^2$': '#b2df8a',
            r'$\Phi$-Div: KL': '#fb9a99',
            r'$\Phi$-Div: $H^2$': '#cab2d6',
            r'$\Phi$-Div: TVD': '#fdbf6f',
            r'PTS': '#01665e',
            #Parametric Mechanisms
            r'MSE$_P$': '#1f78b4',
            r'$\Phi$-Div$_P$: $\chi^2$': '#33a02c',
            r'$\Phi$-Div$_P$: KL': '#e31a1c',
            r'$\Phi$-Div$_P$: $H^2$': '#6a3d9a', 
            r'$\Phi$-Div$_P$: TVD': '#ff7f00',
            
    }

mechanism_marker_map = {
        
            #Non-Parametric Mechanisms
            r'MSE': 'o',
            r'DMI': 'o', 
            r'OA': 'o',
            r'$\Phi$-Div: $\chi^2$': 'o',
            r'$\Phi$-Div: KL': 'o',
            r'$\Phi$-Div: $H^2$': 'o',
            r'$\Phi$-Div: TVD': 'o',
            r'PTS': 'o',
            #Parametric Mechanisms
            r'MSE$_P$': 'P',
            r'$\Phi$-Div$_P$: $\chi^2$': 'P',
            r'$\Phi$-Div$_P$: KL': 'P',
            r'$\Phi$-Div$_P$: $H^2$': 'P', 
            r'$\Phi$-Div$_P$: TVD': 'P',
            
    }

mechanism_dash_map = {
        
            #Non-Parametric Mechanisms
            r'MSE': (1, 1),
            r'DMI': (1, 1), 
            r'OA': (1, 1), 
            r'$\Phi$-Div: $\chi^2$': (1, 1), 
            r'$\Phi$-Div: KL': (1, 1), 
            r'$\Phi$-Div: $H^2$': (1, 1), 
            r'$\Phi$-Div: TVD': (1, 1), 
            r'PTS': (1, 1), 
            #Parametric Mechanisms
            r'MSE$_P$': (5, 5), 
            r'$\Phi$-Div$_P$: $\chi^2$': (5, 5), 
            r'$\Phi$-Div$_P$: KL': (5, 5),
            r'$\Phi$-Div$_P$: $H^2$': (5, 5),
            r'$\Phi$-Div$_P$: TVD': (5, 5),
            
    }

strategy_color_map = {
            "NOISE": '#999999',
            "FIX-BIAS": '#f781bf',
            "MERGE": '#a65628',
            "PRIOR": '#ffff33', 
            "ALL10": '#ff7f00',
            "HEDGE": '#984ea3'
    }

strategy_marker_map = {
            "NOISE": 's',
            "FIX-BIAS": 's',
            "MERGE": 's',
            "PRIOR": 's', 
            "ALL10": 's',
            "HEDGE": 's'
    }

strategy_dash_map = {
            "NOISE": (3, 1, 1, 1, 1, 1),
            "FIX-BIAS": (3, 1, 1, 1, 1, 1),
            "MERGE": (3, 1, 1, 1, 1, 1),
            "PRIOR": (3, 1, 1, 1, 1, 1), 
            "ALL10": (3, 1, 1, 1, 1, 1),
            "HEDGE": (3, 1, 1, 1, 1, 1)
    }

estimation_procedure_map = {
            "Consensus-Grade": r'Consensus Grade',
            "Procedure": r'Procedure',
            "Procedure-NB": r'Procedure-NB'
    }

estimation_procedure_color_map = {
            r'Consensus Grade': '#b2df8a',
            r'Procedure': '#1f78b4',
            r'Procedure-NB': '#a6cee3'
    }

def plot_median_auc(results, filename):
    """
    Used for Binary Effort setting. Generates a lineplot of the median AUC as the number of active graders varies.

    Parameters
    ----------
    results : dict.
        { num: { mechanism: { "Median ROC-AUC": score } } },
        where num is the number of active graders (int), mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), and score is a float.
    filename : str.
        Name of the file used for saving the plot (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    formatted_results = {"Number of Active Graders": [], 
                         "Median AUC": [],
                         "Mechanism": []
                         }
    
    for key in results.keys():
        mechanisms = list(results[key].keys())
        for mechanism in mechanisms:
            formatted_results["Number of Active Graders"].append(key)
            formatted_results["Median AUC"].append(results[key][mechanism]["Median ROC-AUC"])
            formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
            
    results_df = pd.DataFrame(data=formatted_results)
    _ = sns.lineplot(x="Number of Active Graders", y="Median AUC", hue="Mechanism", style="Mechanism", markers=mechanism_marker_map, dashes=mechanism_dash_map, data=results_df, palette=mechanism_color_map)
    
    plt.tight_layout()
    figure_file = "figures/" + filename + ".pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()
    
def plot_auc_scores(results, filename):
    """
    Used for Binary Effort setting. Generates a boxplot of the AUC scores of each mechanism for a fixed number of active graders.

    Parameters
    ----------
    results : dict.
        { mechanism: { "ROC-AUC Scores": [ score ] } },
        where mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map) and each score in the list is a float.
    filename : str.
        Name of the file used for saving the plot (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    formatted_results = {"AUC": [], 
                         "Mechanism": []
                         }
    
    mechanisms = list(results.keys())
    for mechanism in mechanisms:
        for auc in results[mechanism]["ROC-AUC Scores"]:
            formatted_results["AUC"].append(auc)
            formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
            
    results_df = pd.DataFrame(data=formatted_results)
    _ = sns.boxplot(x="Mechanism", y="AUC", data=results_df, palette=mechanism_color_map)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    figure_file = "figures/" + filename + ".pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()
    
def plot_auc_strategic(results, filename):
    """
    Used with strategic agents to compare payments between strategic and truthful agents. 
    For each mechanism, generates boxplots of the AUC scores as the number of strategic agents varies.

    Parameters
    ----------
    results : dict.
        { strategy: { num: { mechanism: { "ROC-AUC Scores": [ score ] } } } },
        where strategy is the name of the strategy (str), num is the number of strategic graders (int), mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), and each score in the list is a float.
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    formatted_results = {"Number of Strategic Graders": [], 
                         "AUC": [],
                         "Mechanism": [],
                         "Strategy": []
                         }
    
    for strategy in results.keys():
        for key in results[strategy].keys():
            if int(key) in [10, 30, 50, 70, 90]:
                mechanisms = list(results[strategy][key].keys())
                for mechanism in mechanisms:
                    for s in results[strategy][key][mechanism]["ROC-AUC Scores"]:
                        formatted_results["Number of Strategic Graders"].append(key)
                        formatted_results["AUC"].append(s)
                        formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
                        formatted_results["Strategy"].append(strategy)
                    
    results_df = pd.DataFrame(data=formatted_results)
    
    for mechanism in mechanisms:
        title = mechanism_name_map[mechanism]
        mechanism_df = results_df.loc[results_df["Mechanism"] == title]
        _ = sns.boxplot(x="Number of Strategic Graders", y="AUC", hue="Strategy", palette=strategy_color_map, data=mechanism_df)
        
        plt.title(title)
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + mechanism + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
    
def plot_estimation_mses(results, filename):
    """
    Used for true grade recovery tests with the various estimation procedures. Generates a boxplot of the MSE scores for each procedure.

    Parameters
    ----------
    results : dict.
        { procedure: { "MSE Scores": [ score ] } },
        where procedure is the name of the procedure (str, one of the keys of the global estimation_procedure_map) and each score in the list is a float.
    filename : str.
        Name of the file used for saving the plot (as a .pdf).

    Returns
    -------
    None.

    """
    global estimation_procedure_map
    
    formatted_results = {
                         "MSE": [],
                         "Estimation Method": []
                         }
    
    mechanisms = list(results.keys())
    for mechanism in mechanisms:
        for m in results[mechanism]["MSE Scores"]:
            formatted_results["MSE"].append(m)
            formatted_results["Estimation Method"].append(estimation_procedure_map[mechanism])
        
    results_df = pd.DataFrame(data=formatted_results)
    _ = sns.boxplot(x="Estimation Method", y="MSE", palette=estimation_procedure_color_map, data=results_df)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    figure_file = "figures/" + filename + ".pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()
    
def plot_kendall_tau(results, filename):
    """
    Used for Continuous Effort setting. Generates a boxplot with the Kendall rank correlation coefficient (tau) scores of each mechanism.

    Parameters
    ----------
    results : dict.
        { mechanism: { "Tau Scores": [ score ] } },
        where mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map) and each score in the list is a float.
    filename : str.
        Name of the file used for saving the plot (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    formatted_results = {"Tau": [], 
                         "Mechanism": []
                         }
    
    mechanisms = list(results.keys())
    for mechanism in mechanisms:
        for t in results[mechanism]["Tau Scores"]:
            formatted_results["Tau"].append(t)
            formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
            
    results_df = pd.DataFrame(data=formatted_results)
    _ = sns.boxplot(x="Mechanism", y="Tau", data=results_df, palette=mechanism_color_map)
    plt.xticks(rotation=45)
    _.set_ylabel(r'$\tau_B$')
    
    plt.tight_layout()
    figure_file = "figures/" + filename + ".pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()
    
def plot_kendall_taus(results, filename):
    """
    Used with strategic agents to compare mechanism performance as the number of strategic agents varies. 
    For each strategy, generates a boxplot with the tau scores of each mechanism as the number of strategic agents varies.

    Parameters
    ----------
    results : dict.
        { strategy: { num : { mechanism: { "Tau Scores": [ score ] } } } },
        where strategy is the name of the strategy (str), num is the number of strategic graders (int), mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), and each score in the list is a float.
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    mechanisms = []
    strategies = []
    
    formatted_results = {"Number of Strategic Graders": [], 
                         "Tau": [],
                         "Mechanism": [],
                         "Strategy": []
                         }
    
    strategies = list(results.keys())
    for strategy in results.keys():
        for key in results[strategy].keys():
            if int(key) in [0, 20, 40, 60, 80, 100]: 
                mechanisms = list(results[strategy][key].keys())
                for mechanism in mechanisms:
                    for t in results[strategy][key][mechanism]["Tau Scores"]:
                        formatted_results["Number of Strategic Graders"].append(key)
                        formatted_results["Tau"].append(t)
                        formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
                        formatted_results["Strategy"].append(strategy)
            
    results_df = pd.DataFrame(data=formatted_results)
    
    for strategy in strategies:
        title = strategy
        strategy_df = results_df.loc[results_df["Strategy"] == title]
        ax = sns.boxplot(x="Number of Strategic Graders", y="Tau", hue="Mechanism", palette=mechanism_color_map, data=strategy_df)
        ax.set_ylabel(r'$\tau_B$')
        
        plt.title(title)
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + strategy + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
    
def plot_kendall_tau_variances(results, filename):
    """
    Used for Continuous Effort setting. Generates a boxplot with the variances of the Kendall rank correlation coefficient (tau) scores of each mechanism.

    Parameters
    ----------
    results : dict.
        { mechanism: { "Tau Variances": [ score ] } },
        where mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map) and each score in the list is a float.
    filename : str.
        Name of the file used for saving the plot (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    formatted_results = {"Variance": [], 
                         "Mechanism": []
                         }
    
    mechanisms = list(results.keys())
    for mechanism in mechanisms:
        for v in results[mechanism]["Tau Variances"]:
            formatted_results["Variance"].append(v)
            formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
            
    results_df = pd.DataFrame(data=formatted_results)
    ax = sns.boxplot(x="Mechanism", y="Variance", data=results_df, palette=mechanism_color_map)
    plt.xticks(rotation=45)
    ax.set_ylabel(r'Variance of $\tau_B$')
    
    plt.tight_layout()
    figure_file = "figures/" + filename + ".pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()
    
def plot_mean_rank_changes(results, filename):
    """
    Used with strategic agents to see how agents can benefit from deviating from truthful to strategic reporting. 
    For each mechanism, generates a lineplot with the average gain in payment rank of a single deviating agent as the number of strategic agents varies.

    Parameters
    ----------
    results : dict.
        { strategy: { num : { mechanism: { "Mean Gain": score } } } },
        where strategy is the name of the strategy (str), num is the number of strategic graders (int), mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), and score is the mean rank gain (float).
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    mechanisms = []
    
    formatted_results = {"Number of Strategic Graders": [], 
                         "Mean Rank Gain": [],
                         "Mechanism": [],
                         "Strategy": []
                         }
    
    for strategy in results.keys():
        for key in results[strategy].keys():
            mechanisms = list(results[strategy][key].keys())
            for mechanism in mechanisms:
                formatted_results["Number of Strategic Graders"].append(key)
                formatted_results["Mean Rank Gain"].append(results[strategy][key][mechanism]["Mean Gain"])
                formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
                formatted_results["Strategy"].append(strategy)
            
    results_df = pd.DataFrame(data=formatted_results)
    
    
    for mechanism in mechanisms:
        title = mechanism_name_map[mechanism]
        mechanism_df = results_df.loc[results_df["Mechanism"] == title]
        _ = sns.lineplot(x="Number of Strategic Graders", y="Mean Rank Gain", hue="Strategy", style="Strategy", palette=strategy_color_map, markers=strategy_marker_map, dashes=strategy_dash_map, data=mechanism_df)
        
        plt.title(title)
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + mechanism + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()
        
def plot_variance_rank_changes(results, filename):
    """
    Used with strategic agents to see the variance of how much agents can benefit from deviating from truthful to strategic reporting. 
    For each mechanism, generates a lineplot with the variance of the gain in payment rank of a single deviating agent as the number of strategic agents varies.

    Parameters
    ----------
    results : dict.
        { strategy: { num : { mechanism: { "Rank Gain Variance": score } } } },
        where strategy is the name of the strategy (str), num is the number of strategic graders (int), mechanism is the name of a mechanism (str, one of the keys of the global mechanism_name_map), and score is the variance of the rank gain (float).
    filename : str.
        Name of the file prefix used for saving the plots (as a .pdf).

    Returns
    -------
    None.

    """
    global mechanism_name_map
    
    mechanisms = []
    
    formatted_results = {"Number of Strategic Graders": [], 
                         "Rank Gain Variance" : [],
                         "Mechanism": [],
                         "Strategy": []
                         }
    
    for strategy in results.keys():
        for key in results[strategy].keys():
            mechanisms = list(results[strategy][key].keys())
            for mechanism in mechanisms:
                formatted_results["Number of Strategic Graders"].append(key)
                formatted_results["Rank Gain Variance"].append(results[strategy][key][mechanism]["Variance Gain"])
                formatted_results["Mechanism"].append(mechanism_name_map[mechanism])
                formatted_results["Strategy"].append(strategy)
            
    results_df = pd.DataFrame(data=formatted_results)
    
    
    for mechanism in mechanisms:
        title = mechanism_name_map[mechanism]
        mechanism_df = results_df.loc[results_df["Mechanism"] == title]
        _ = sns.lineplot(x="Number of Strategic Graders", y="Rank Gain Variance", hue="Strategy", style="Strategy", palette=strategy_color_map, markers=strategy_marker_map, dashes=strategy_dash_map, data=mechanism_df)
        
        plt.title(title)
        plt.tight_layout()
        figure_file = "figures/" + filename + "-" + mechanism + ".pdf"
        plt.savefig(figure_file, dpi=300)
        plt.show()
        plt.close()