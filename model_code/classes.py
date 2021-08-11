"""
Class definitions for the fundamental objects in the agent-based model.

@author: Noah Burrell <burrelln@umich.edu>
"""

from scipy.stats import binom, norm, halfnorm, uniform

class Student:
    """
    A (truthfully-reporting) Student object.
    
    Attributes
    ----------
    id : int.
         Unique identifier.
    type : str "active" or "passive". 
           Denotes effort level for binary effort settings.
    lam : float.
          Denotes effort level for continuous effort settings.
    bias : float.
           Denotes bias when grading submissions.
    payment : int or float (depending on the mechanism).
              Stores payments received over the course of a semester.
    grades : dict.
             Stores the grades assigned over the course of a semester.
             grades = { assignment_number (int): { submission_number (int): score (int 0-10)} }
    """
    
    def __init__(self, num, grader_type="active"):
        """
        Creates a Student object.
        
        Parameters
        ----------
        num : int.
              Identification number.
        grader_type : str "active" or "passive".

        """
        self.id = num
        self.type = grader_type
        self.payment = 0     
        
        self.bias = norm.rvs(loc=0, scale=1, random_state=None)
    
        lam = 0
        while lam == 0:
            lam = uniform.rvs(loc=0, scale=2, random_state=None)
        self.lam = lam
        
        self.grades = {}
        
    def report(self, signal):
        """
        Generates a report for the Student object given a signal. 
        
        Parameters
        ----------
        signal : int 0-10.
                 Signal observed by the Student grading a submission.
        
        Returns
        -------
        report : int 0-10.
                 Equal to signal, since Student objects report truthfully.
        """
        report = signal
        return report
        
class StrategicStudent(Student):
    """
    A Student object that reports strategically. 

    Attributes
    ----------
    Inherits all attributes from Student class.
    
    Additional attributes:
        strategy : str one of the following:
                    - "TRUTH" (with this strategy, StrategicStudents behave just like regular Students)
                    - "NOISE"
                    - "FIX-BIAS"
                    - "MERGE"
                    - "PRIOR"
                    - "ALL10"
                    - "HEDGE"
                  A strategy to follow. See the paper for a description of each strategy.
            
            
        bias_correction : float.
                          Denotes a bias correction term that is accessed when using the "Fix-Bias" strategy.
    """
    
    def __init__(self, num, strat="TRUTH"):
        """
        Creates a StrategicStudent object.
        
        Parameters
        ----------
        num :  int.
               Identification number.
        strat : str.
                A strategy to follow (from the list above).
        
        """
        super().__init__(num)

        self.strategy = strat
        
        bias_correction_magnitude = halfnorm.rvs(loc=0, scale=1, random_state=None)
        bias_correction_sign = -1
        if self.bias < 0:
            bias_correction_sign = 1
            
        self.bias_correction = bias_correction_sign * bias_correction_magnitude
    
        
    def report(self, signal):
        """
        Generates a report for the StrategicStudent object given a signal. Supersedes report() method from Student class.
        
        Parameters
        ----------
        signal : int 0-10.
                 Signal observed by the StrategicStudent grading a submission.
        
        Returns
        -------
        report : int 0-10.
                 Output of applying the StrategicStudent's given strategy to the signal.
                 Note---if the strategy attribute does not match one of the strategies from the list above, 
                       this function just returns report = signal, just as for Student objects and StrategicStudent objects with the strategy "TRUTH".
        """
        sigma = self.strategy
        
        if sigma == "NOISE":
            noise = norm.rvs(loc=0, scale=1, random_state=None)
            noisy_signal = signal + noise
            report = int(round(noisy_signal))
            if report > 10:
                report = 10
            
            elif report < 0:
                report = 0
            
        
        elif sigma == "FIX-BIAS":
            corrected_signal = signal + self.bias_correction
            report = int(round(corrected_signal))
            if report > 10:
                report = 10
            
            elif report < 0:
                report = 0
            
        elif sigma == "MERGE":
            projection = {
                    0: 0,
                    1: 3,
                    2: 3,
                    3: 3,
                    4: 6,
                    5: 6,
                    6: 6,
                    7: 7,
                    8: 7,
                    9: 7,
                    10: 10
                }
            
            report = projection[signal]
            
        elif sigma == "PRIOR":
            report = 7
            
        elif sigma == "ALL10":
            report = 10
        
        elif sigma == "HEDGE":
            posterior = (7 + signal)/2.0
            report = int(round(posterior))
        
        else: 
            report = signal
            
        return report
        
    
class Submission:
    """
    A Submission object.

    Attributes
    ----------
    student_id : int.
                 Unique identifier for the submission; corresponds to id number of Student who "turned in" this submission for the given assignment. 
    assignment_number : int.
                        Assignment identifier. Semesters consist of one or more assignments; the meta-grading mechanism is applied sequentially over the course of a semester, once for each assignment, to calculate payments to the peer grading agents.
    true_grade : int 0-10.
                 The ground truth score for the submission.
    grades : dict.
             Stores the reports from each Student who graded this submission.
             grades = {grader id (int): score (int 0-10) } 
    """
    def __init__(self, s_id, assignment_num):
        """
        Creates a Submission object.
        
        Parameters
        ----------
        s_id : int submission identification number.
        assignment_num : int assignment identification number.

        """
        self.student_id = s_id
        self.assignment_number = assignment_num
        self.true_grade = binom.rvs(n=10, p=0.7, random_state=None)
        
        self.grades = {}
