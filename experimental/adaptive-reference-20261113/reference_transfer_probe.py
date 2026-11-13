"""Extra-information diagnostic, not a two-image inference engine."""
import numpy as np
from capacity_audit import ridge_fit


def crossfit_transfer(reference_features,reference_values,target_features,mask,ridge=1e-5):
    """Fit only reference pairs; target truth is deliberately absent."""
    source=np.asarray(reference_features,np.float64);values=np.asarray(reference_values,np.float64);target=np.asarray(target_features,np.float64)
    if source.ndim!=2 or values.ndim!=2 or values.shape!=(source.shape[0],3) or target.ndim!=2 or target.shape[1]!=source.shape[1] or target.shape[0]<1 or not np.isfinite(target).all():
        raise ValueError('Invalid reference/target design')
    predictions=np.empty_like(values);parameters=[]
    for train in [mask,~mask]:
        coefficients=ridge_fit(source,values,train,ridge=ridge)
        predictions[~train]=source[~train]@coefficients
        parameters.append(coefficients)
    mean=np.mean(parameters,axis=0);transferred=target@mean
    if not np.isfinite(transferred).all():raise ValueError('Nonfinite transferred output')
    return dict(coefficients=mean,fold_coefficients=parameters,reference_crossfit=predictions,target_prediction=transferred)
