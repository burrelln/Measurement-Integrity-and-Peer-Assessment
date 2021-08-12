"""
Recreate the figures in the paper by uncommenting the associated function at the bottom of this script.

The generated figures are saved in either the "main_paper" or "appendix" folders in the "figures" folder located in the "data" directory (i.e. the same directory as this file).

Note the numbering of the figures corresponds with the full version of the paper (on arXiv).

@author: Noah Burrell <burrelln@umich.edu>
"""
from json import load
import os
from statistics import mean
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib.pyplot as plt  
import seaborn as sns

from model_code import *

def figure_1():
    old_names = {value:key for key,value in mechanism_name_map.items()}
        
    data = {
            "Mechanism": 
                [
                    # Non-Parametric Mechanisms
                    r'MSE',
                    r'DMI', 
                    r'OA',
                    r'$\Phi$-Div: $\chi^2$',
                    r'$\Phi$-Div: KL',
                    r'$\Phi$-Div: $H^2$',
                    r'$\Phi$-Div: TVD',
                    r'PTS',
                    # Parametric Mechanisms
                    r'MSE$_P$',
                    r'$\Phi$-Div$_P$: $\chi^2$',
                    r'$\Phi$-Div$_P$: KL',
                    r'$\Phi$-Div$_P$: $H^2$',
                    r'$\Phi$-Div$_P$: TVD'
                ],
            "Theoretical Guarantee":
                [
                    # Non-Parametric Mechanisms
                    r'None',
                    r'Dominantly Truthful',
                    r'Truthful',             
                    r'Strongly Truthful',  
                    r'Strongly Truthful',  
                    r'Strongly Truthful',  
                    r'Strongly Truthful',   
                    r'Helpful Reporting',
                    # Parametric Mechanisms
                    r'None',
                    r'Strongly Truthful',  
                    r'Strongly Truthful',  
                    r'Strongly Truthful',  
                    r'Strongly Truthful',
                ]
        }
        
    guarantee_value_map = {
            "None": 0,
            "Helpful Reporting": 1,
            "Truthful": 2,
            "Informed Truthful": 3,
            "Strongly Truthful": 4,
            "Dominantly Truthful": 5
        }
    
    cm = sns.color_palette("magma")
    guarantee_color_map = {key:cm[i] for i, key in enumerate(guarantee_value_map.keys())}
    
    json_data = {}
    filename = 'be-no_bias-best_mechanisms'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        json_data.update(d)
        
    filename = 'be-no_bias-nonparam'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for num in range(10, 100, 10):
            json_data[str(num)].update(d[str(num)])
        
    filename = 'be-no_bias-phi_div'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for num in range(10, 100, 10):
            json_data[str(num)].update(d[str(num)])
        
    filename = 'be-no_bias-phi_div_p'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for num in range(10, 100, 10):
            json_data[str(num)].update(d[str(num)])
    
    be_accuracy_list = []
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        values = [json_data[str(num)][key]["Median ROC-AUC"] for num in range(10, 100, 10)]
        value = mean(values)
        be_accuracy_list.append(value)
        
    data["Accuracy"] = be_accuracy_list
    
    robustness_list = []
    strategies = [
                    "NOISE",
                    "FIX-BIAS",
                    "MERGE",
                    "PRIOR", 
                    "ALL10", 
                    "HEDGE"
                ]
    
    json_data = {}
        
    filename = 'incentives_for_deviating-ce-bias'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        json_data.update(d)
    
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        values = [-1*json_data[strategy][str(num)][key]["Mean Gain"] for strategy in strategies for num in range(10, 100, 10)]
        value = mean(values)
        robustness_list.append(value)
        
    data["Robustness"] = robustness_list
    
    """
    # This code block can be uncommented to set up (with a few additional adjustments below) 
    # an analogous plot considering only a single strategy, e.g. PRIOR as below, 
    # instead of aggregating performance over all considered strategies.
    
    robustness_list_prior = []
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        value = -1*json_data["PRIOR"]["50"][key]["Mean Gain"]
        robustness_list_prior.append(value)
        
    data["Robustness (Prior)"] = robustness_list_prior
    """
    
    _ = sns.scatterplot(x="Robustness", y="Accuracy", hue="Theoretical Guarantee", palette=guarantee_color_map, data=data)
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # order = [0,4,2,3,5,1] # When Informed Truthful is included (6 labels)
    order=[0,4,2,3,1] # When Informed Truthful is not included
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], title=r'Theoretical Guarantee')
    
    # label points on the plot
    for x, y, s in zip(data["Robustness"], data["Accuracy"], data["Mechanism"]):
        if s == r'$\Phi$-Div: $H^2$' or s == r'MSE' or s == r'MSE$_P$':
            x_val = x - 2
            y_val = y - 0.025
            
        elif s == r'$\Phi$-Div$_P$: TVD':
            x_val = x - 6.5
            y_val = y + 0.015
            
        elif s == r'$\Phi$-Div$_P$: KL' or s == r'$\Phi$-Div$_P$: $\chi^2$':
            x_val = x - 2.5
            y_val = y + 0.015
            
        else:
            x_val = x - 2
            y_val = y + 0.015
        
            
        plt.text(x = x_val, # x-coordinate position of data label
        y = y_val, # y-coordinate position of data label
        s = s, # data label
        color = 'black') # set colour of line     
    
    ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    ax.set_facecolor('gainsboro')
    
    ax.set_ylabel(r'Measurement Integrity')
    ax.set_xlabel(r'Robustness Against Strategic Behavior')
    plt.title(r'Apparent Trade-off Between Integrity and Robustness')
    plt.tight_layout()
    filename = "2D-tradeoff"
    figure_file = "figures/main_paper/" + filename + ".pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()

def figure_2a():
    filename = 'be-no_bias-phi_div-box'
    json_file = filename + '.json'
    pdf_file = 'main_paper/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_auc_scores(data, pdf_file)

def figure_2b():
    filename = 'be-no_bias-phi_div'
    json_file = filename + '.json'
    pdf_file = 'main_paper/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_median_auc(data, pdf_file)
    
def figure_2c():
    filename = 'be-no_bias-nonparam-box'
    json_file = filename + '.json'
    pdf_file = 'main_paper/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_auc_scores(data, pdf_file)

def figure_2d():
    filename = 'be-no_bias-nonparam'
    json_file = filename + '.json'
    pdf_file = 'main_paper/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_median_auc(data, pdf_file)
    
def figure_2e():
    filename = 'be-no_bias-best_mechanisms-box'
    json_file = filename + '.json'
    pdf_file = 'main_paper/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_auc_scores(data, pdf_file)

def figure_2f():
    filename = 'be-no_bias-best_mechanisms'
    json_file = filename + '.json'
    pdf_file = 'main_paper/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_median_auc(data, pdf_file)
    
def figure_3a():
    filename = 'be-bias-param_and_baseline-box'
    json_file = filename + '.json'
    pdf_file = 'main_paper/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_auc_scores(data, pdf_file)

def figure_3b():
    filename = 'be-bias-param_and_baseline'
    json_file = filename + '.json'
    pdf_file = 'main_paper/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_median_auc(data, pdf_file)
    
def figure_4a():
    filename = 'ce-bias-param'
    json_file = filename + '.json'
    pdf_file = 'main_paper/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_kendall_tau(data, pdf_file)

def figure_4b():
    filename = 'ce-bias-variance_of_ranking_quality-param'
    json_file = filename + '.json'
    pdf_file = 'main_paper/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_kendall_tau_variances(data, pdf_file)

def figures_5_and_E_2a_and_E_2b():
    """
    Note that this produces figures for both the main paper (Figure 5) and the appendix (Figure E.2a and E.2b).
    However, due to the nature of the plotting functions, the recreated figures are all saved in the "figures/main_paper" directory. 
    """
    filename = 'incentives_for_deviating-ce-bias'
    json_file = filename + '.json'
    
    pdf_file = 'main_paper/' + filename + "-mean_gain"
    with open(json_file,"r") as file:
        data = load(file)
    plot_mean_rank_changes(data, pdf_file)
    
    pdf_file = 'main_paper/' + filename + "-variance_gain"
    with open(json_file,"r") as file:
        data = load(file)
    plot_variance_rank_changes(data, pdf_file)
    
def figures_6_and_E_2c():
    """
    Note that this produces figures for both the main paper (Figure 6) and the appendix (Figure E.2c).
    However, due to the nature of the plotting functions, the recreated figures are all saved in the "figures/main_paper" directory. 
    """
    filename = 'truthful_vs_strategic_payments-ce-bias'
    json_file = filename + '.json'
    
    pdf_file = 'main_paper/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_auc_strategic(data, pdf_file)

def figures_7_and_E_2d():
    """
    Note that this produces figures for both the main paper (Figure 7) and the appendix (Figure E.2d).
    However, due to the nature of the plotting functions, the recreated figures are all saved in the "figures/main_paper" directory. 
    """
    filename = 'strategic-ce-bias'
    json_file = filename + '.json'
    pdf_file = 'main_paper/' + filename
    
    with open(json_file,"r") as file:
        data = load(file)
    plot_kendall_taus(data, pdf_file)
    
def figure_D_1a():
    old_names = {value:key for key,value in mechanism_name_map.items()}
    old_names.update(
            {y:x for x, y in {
                "AMSE_P: 0": r'AMSE$_P$',
                "CORR: 0": r'CORR',
                "MCC: 0": r'MCC',
                "Phi-DIV_P*: CHI_SQUARED": r'$\Phi$-Div$_{P}^*$: $\chi^2$',
                "Phi-DIV_P*: KL": r'$\Phi$-Div$_{P}^*$: KL',
                "Phi-DIV_P*: SQUARED_HELLINGER": r'$\Phi$-Div$_{P}^*$: $H^2$', 
                "Phi-DIV_P*: TVD": r'$\Phi$-Div$_{P}^*$: TVD',
                "R2: 0": r'$R^2$',
                }.items()
            }
        )
        
    data = {
            "Mechanism": 
                [
                    # Non-Parametric Mechanisms
                    r'MSE',
                    r'DMI', 
                    r'OA',
                    r'$\Phi$-Div: $\chi^2$',
                    r'$\Phi$-Div: KL',
                    r'$\Phi$-Div: $H^2$',
                    r'$\Phi$-Div: TVD',
                    r'PTS',
                    # Parametric Mechanisms
                    r'MSE$_P$',
                    r'$\Phi$-Div$_P$: $\chi^2$',
                    r'$\Phi$-Div$_P$: KL',
                    r'$\Phi$-Div$_P$: $H^2$',
                    r'$\Phi$-Div$_P$: TVD',
                    # Extensions
                    r'AMSE$_P$',
                    r'CORR',
                    r'MCC',
                    r'$\Phi$-Div$_{P}^*$: $\chi^2$',
                    r'$\Phi$-Div$_{P}^*$: KL',
                    r'$\Phi$-Div$_{P}^*$: $H^2$', 
                    r'$\Phi$-Div$_{P}^*$: TVD',
                    r'$R^2$'
                    ],
                "Mechanism Status":
                [
                    # Non-Parametric Mechanisms
                    r'Established',
                    r'Established',
                    r'Established',
                    r'Established',
                    r'Established',
                    r'Established',
                    r'Established',
                    r'Established',
                    # Parametric Mechanisms
                    r'Established',
                    r'Established',
                    r'Established',
                    r'Established',
                    r'Established',
                    # Extensions
                    r'Novel',
                    r'Novel',
                    r'Novel',
                    r'Novel',
                    r'Novel',
                    r'Novel',
                    r'Novel',
                    r'Novel'
                ]
        }
        
    guarantee_value_map = {
            "None": 0,
            "Helpful Reporting": 1,
            "Truthful": 2,
            "Informed Truthful": 3,
            "Strongly Truthful": 4,
            "Dominantly Truthful": 5
        }
    
    cm = sns.color_palette("magma")
    guarantee_color_map = {key:cm[i] for i, key in enumerate(guarantee_value_map.keys())}
    
    json_data = {}
    filename = 'be-no_bias-best_mechanisms'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        json_data.update(d)
        
    filename = 'be-no_bias-nonparam'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for num in range(10, 100, 10):
            json_data[str(num)].update(d[str(num)])
        
    filename = 'be-no_bias-phi_div'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for num in range(10, 100, 10):
            json_data[str(num)].update(d[str(num)])
        
    filename = 'be-no_bias-phi_div_p'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for num in range(10, 100, 10):
            json_data[str(num)].update(d[str(num)])
            
    filename = 'be-no_bias-extensions'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for num in range(10, 100, 10):
            json_data[str(num)].update(d[str(num)])
    
    be_accuracy_list = []
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        values = [json_data[str(num)][key]["Median ROC-AUC"] for num in range(10, 100, 10)]
        value = mean(values)
        be_accuracy_list.append(value)
        
    data["Accuracy"] = be_accuracy_list
    
    robustness_list = []
    strategies = [
                    "NOISE",
                    "FIX-BIAS",
                    "MERGE",
                    "PRIOR", 
                    "ALL10", 
                    "HEDGE"
                ]
    
    json_data = {}
        
    filename = 'incentives_for_deviating-ce-bias'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        json_data.update(d)
        
    filename = 'incentives_for_deviating-ce-bias-extensions'
    json_file = filename + '.json'
    with open(json_file,"r") as file:
        d = load(file)
        for strat in strategies:
            for num in range(10, 100, 10):
                json_data[strat][str(num)].update(d[strat][str(num)])
    
    for mechanism in data["Mechanism"]:
        key = old_names[mechanism]
        values = [-1*json_data[strategy][str(num)][key]["Mean Gain"] for strategy in strategies for num in range(10, 100, 10)]
        value = mean(values)
        robustness_list.append(value)
        
    data["Robustness"] = robustness_list
    
    _ = sns.scatterplot(x="Robustness", y="Accuracy", hue="Mechanism Status", data=data)
    handles, labels = plt.gca().get_legend_handles_labels()
    
    order=[0,1] 
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=9,title=r'Mechanism Status')
    
    # label points on the plot
    for x, y, s in zip(data["Robustness"], data["Accuracy"], data["Mechanism"]):
        if s == r'$\Phi$-Div: $H^2$' or s == r'MCC' or s == r'AMSE$_P$':
            x_val = x - 2
            y_val = y - 0.025
            
        elif s == r'$\Phi$-Div$_P$: TVD':
            x_val = x - 6.5
            y_val = y + 0.015
            
        elif s == r'$\Phi$-Div$_P$: KL' or s == r'$\Phi$-Div$_{P}^*$: KL' or s == r'$\Phi$-Div$_{P}^*$: TVD':
            x_val = x - 2.5
            y_val = y + 0.015
            
        elif s == r'$\Phi$-Div$_{P}^*$: $\chi^2$':
            x_val = x - 8.5
            y_val = y - 0.005
            
        elif s == r'$\Phi$-Div$_P$: $\chi^2$' or s == r'$\Phi$-Div$_{P}^*$: $H^2$':
            x_val = x - 2.5
            y_val = y - 0.03
            
        else:
            x_val = x - 2
            y_val = y + 0.015
            
        plt.text(x = x_val, # x-coordinate position of data label
        y = y_val, # y-coordinate position of data label
        s = s, # data label
        color = 'black') # set colour of line
    
    ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    ax.set_facecolor('gainsboro')
    
    ax.set_ylabel(r'Measurement Integrity')
    ax.set_xlabel(r'Robustness Against Strategic Behavior')
    plt.title(r'Apparent Trade-off Between Integrity and Robustness')
    plt.tight_layout()
    filename = "2D-extensions"
    figure_file = "figures/appendix/" + filename + ".pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()
    
def figure_E_1a():
    filename = 'be-no_bias-phi_div_p-box'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_auc_scores(data, pdf_file)
    
    filename = 'be-no_bias-phi_div_p'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_median_auc(data, pdf_file)

def figure_E_1b():
    filename = 'be-bias-nonparam-box'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_auc_scores(data, pdf_file)
    
    filename = 'be-bias-nonparam'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_median_auc(data, pdf_file)

def figure_E_1c():
    filename = 'ce-bias-nonparam'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_kendall_tau(data, pdf_file)
    
    filename = 'ce-bias-variance_of_ranking_quality-nonparam'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_kendall_tau_variances(data, pdf_file)

def figure_E_3a():
    filename = 'true-grade-recovery_no-bias'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_estimation_mses(data, pdf_file)
    
    filename = 'true-grade-recovery'
    json_file = filename + '.json'
    pdf_file = 'appendix/' + filename
    with open(json_file,"r") as file:
        data = load(file)
    plot_estimation_mses(data, pdf_file)

if __name__ == "__main__":
    """Uncomment a function below to create a .pdf of the associated figure from the paper."""

    #figure_1()
    
    #figure_2a()
    #figure_2b()
    #figure_2c()
    #figure_2d()
    #figure_2e()
    #figure_2f()
    
    #figure_3a()
    #figure_3b()
    #figure_4a()
    #figure_4b()
    
    #figures_5_and_E_2a_and_E_2b()
    #figures_6_and_E_2c()
    #figures_7_and_E_2d()
    
    #figure_D_1a()
    
    #figure_E_1a()
    #figure_E_1b()
    #figure_E_1c()
    
    #figure_E_3a()