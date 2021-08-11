"""
Evaluation metrics that are used to measure the performance of the mechanisms at various tasks.

@author: Noah Burrell <burrelln@umich.edu>
"""

from numpy import isnan
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import kendalltau
from sys import maxsize

def roc_auc(student_list):
    """
    Computes the ROC AUC score for classifying agents as "active" or "passive" based on their payments.
    Payments were assigned according to some mechanism for completing peer grading tasks over the course of a simulated semester.

    Parameters
    ----------
    student_list : list of Student objects.

    Returns
    -------
    score : float. 
            ROC AUC score.

    """
    true = []
    scores = []
    
    minsize = -maxsize - 1
    
    for student in student_list:
        classification = 0
        if student.type == "active":
            classification = 1
        true.append(classification)
        
        p = student.payment
        
        """
        Formatting to ensure all payments are valid inputs to the roc_auc_score function from sklearn.metrics.
        """
        if p > maxsize:
            p = maxsize
        if p < minsize:
            p = minsize
        if isnan(p):
            #print("Payment is nan")
            p = 0
        
        scores.append(p)
        
    score = roc_auc_score(true, scores)
    return score

def roc_auc_strategic(student_list):
    """
    Computes the ROC AUC score for classifying agents as truthful (strategy="TRUTH") or strategic (strategy="NOISE", "FIX-BIAS", "MERGE", "PRIOR", "ALL10", or "HEDGE") based on their payments.
    Payments were assigned according to some mechanism for completing peer grading tasks over the course of a simulated semester.

    Parameters
    ----------
    student_list : list of StrategicStudent objects.

    Returns
    -------
    score : float.
            ROC AUC score.

    """
    true = []
    scores = []
    
    minsize = -maxsize - 1
    
    for student in student_list:
        classification = 0
        if student.strategy == "TRUTH":
            classification = 1
        true.append(classification)
        
        p = student.payment
        
        """
        Formatting to ensure all payments are valid inputs to the roc_auc_score function from sklearn.metrics.
        """
        if p > maxsize:
            p = maxsize
        if p < minsize:
            p = minsize
        if isnan(p):
            #print("Payment is nan")
            p = 0
        
        scores.append(p)
        
    score = roc_auc_score(true, scores)
    return score
    

def kendall_tau(student_list):
    """
    Computes the Kendall rank correlation coefficient (Kendall's tau_B) between the ranking of agents according to the continuous effort parameter (lam) and the ranking of agents according to their payments.
    Payments were assigned according to some mechanism for completing peer grading tasks over the course of a simulated semester.
                                                       
    Parameters
    ----------
    student_list : A list of Student objects.
        
    Returns
    -------
    tau : float.
          Kendall rank correlation coefficient.

    """
    tau = 0 
    
    true = []
    scores = []
    
    minsize = -maxsize - 1
    
    for student in student_list:
        score = student.lam
        true.append(score)
        
        p = student.payment
        
        """
        Formatting to ensure all payments are valid inputs to the kendalltau function from scipy.stats.
        """
        if p > maxsize:
            p = maxsize
        if p < minsize:
            p = minsize
        if isnan(p):
            #print("Payment is nan")
            p = 0
        
        scores.append(p)
    
    tau, p_value = kendalltau(true, scores)
    
    return tau

def true_grade_mse(true_scores, computed_scores):
    """
    Computes the mean squared error (MSE) of estimates of the ground truth scores for some submissions.

    Parameters
    ----------
    true_scores : list of int 0-10. 
                  The ground truth scores for the submissions.
    computed_scores : list of float. 
                      The estimated scores for the submissions.

    Returns
    -------
    mse : float.
          The mean squared error of the computed scores.

    """
    return mean_squared_error(true_scores, computed_scores)