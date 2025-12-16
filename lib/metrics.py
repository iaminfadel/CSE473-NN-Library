"""
Metrics module for evaluating machine learning models.

This module provides basic evaluation metrics for classification tasks,
serving as a lightweight alternative to sklearn.metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Union


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted labels.
    
    Returns:
    --------
    accuracy : float
        Accuracy score between 0 and 1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List] = None) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        List of labels to index the matrix.
    
    Returns:
    --------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    n_labels = len(labels)
    label_to_ind = {label: i for i, label in enumerate(labels)}
    
    # Initialize confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_ind[true_label]
        pred_idx = label_to_ind[pred_label]
        cm[true_idx, pred_idx] += 1
    
    return cm


def precision_recall_fscore_support(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    labels: Optional[List] = None,
    average: Optional[str] = None
) -> tuple:
    """
    Compute precision, recall, F-measure and support for each class.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        List of labels to index the matrix.
    average : str, optional
        Type of averaging performed on the data.
    
    Returns:
    --------
    precision : ndarray or float
        Precision scores.
    recall : ndarray or float
        Recall scores.
    fscore : ndarray or float
        F1 scores.
    support : ndarray or int
        Support (number of occurrences) for each class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(y_true)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels)
    
    # Calculate metrics for each class
    precision = np.zeros(len(labels))
    recall = np.zeros(len(labels))
    fscore = np.zeros(len(labels))
    support = np.zeros(len(labels), dtype=int)
    
    for i, label in enumerate(labels):
        tp = cm[i, i]  # True positives
        fp = np.sum(cm[:, i]) - tp  # False positives
        fn = np.sum(cm[i, :]) - tp  # False negatives
        
        # Precision
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-score
        if precision[i] + recall[i] > 0:
            fscore[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            fscore[i] = 0.0
        
        # Support
        support[i] = np.sum(cm[i, :])
    
    if average == 'macro':
        return (np.mean(precision), np.mean(recall), np.mean(fscore), np.sum(support))
    elif average == 'weighted':
        weights = support / np.sum(support)
        return (
            np.average(precision, weights=weights),
            np.average(recall, weights=weights),
            np.average(fscore, weights=weights),
            np.sum(support)
        )
    else:
        return precision, recall, fscore, support


def classification_report(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None,
    output_dict: bool = False
) -> Union[str, Dict]:
    """
    Build a text report showing the main classification metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        List of labels to index the matrix.
    target_names : list of str, optional
        Display names matching the labels.
    output_dict : bool, default=False
        If True, return output as dict.
    
    Returns:
    --------
    report : str or dict
        Classification report.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(y_true)
    
    if target_names is None:
        target_names = [str(label) for label in labels]
    
    # Compute metrics
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, labels)
    
    # Compute averages
    macro_avg = precision_recall_fscore_support(y_true, y_pred, labels, average='macro')
    weighted_avg = precision_recall_fscore_support(y_true, y_pred, labels, average='weighted')
    
    if output_dict:
        report_dict = {}
        
        # Per-class metrics
        for i, (label, name) in enumerate(zip(labels, target_names)):
            report_dict[name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1-score': fscore[i],
                'support': int(support[i])
            }
        
        # Averages
        report_dict['macro avg'] = {
            'precision': macro_avg[0],
            'recall': macro_avg[1],
            'f1-score': macro_avg[2],
            'support': int(macro_avg[3])
        }
        
        report_dict['weighted avg'] = {
            'precision': weighted_avg[0],
            'recall': weighted_avg[1],
            'f1-score': weighted_avg[2],
            'support': int(weighted_avg[3])
        }
        
        # Overall accuracy
        report_dict['accuracy'] = accuracy_score(y_true, y_pred)
        
        return report_dict
    
    else:
        # Format as string
        width = max(len(name) for name in target_names)
        width = max(width, len('weighted avg'))
        
        headers = ['precision', 'recall', 'f1-score', 'support']
        fmt = f'{{:>{width}s}} ' + ' '.join(['{:>9s}'] * len(headers))
        
        report = fmt.format('', *headers) + '\n\n'
        
        # Per-class metrics
        for i, name in enumerate(target_names):
            values = [f'{precision[i]:.2f}', f'{recall[i]:.2f}', f'{fscore[i]:.2f}', f'{int(support[i])}']
            report += fmt.format(name, *values) + '\n'
        
        report += '\n'
        
        # Averages
        values = [f'{macro_avg[0]:.2f}', f'{macro_avg[1]:.2f}', f'{macro_avg[2]:.2f}', f'{int(macro_avg[3])}']
        report += fmt.format('macro avg', *values) + '\n'
        
        values = [f'{weighted_avg[0]:.2f}', f'{weighted_avg[1]:.2f}', f'{weighted_avg[2]:.2f}', f'{int(weighted_avg[3])}']
        report += fmt.format('weighted avg', *values) + '\n'
        
        return report