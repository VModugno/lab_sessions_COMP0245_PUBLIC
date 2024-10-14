
import numpy as np
import os
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
from skopt import gp_minimize
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting


def fit_gp_model_1d(X_values, y_values):
    # Define the kernel
    rbf_kernel = RBF(
    length_scale=1.0,            # Initial length scale
    length_scale_bounds=(1e-2, 1e2)  # Bounds for length scale
    )
    
    # Create and fit the GP regressor
    gp = GaussianProcessRegressor(
        kernel=rbf_kernel,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=42
    )
    gp.fit(X_values, y_values)
    
    return gp

def create_prediction_range_kp0():
    kp0_range = np.linspace(0.1, 1000, 200).reshape(-1, 1)
    return kp0_range

def create_prediction_range_kd0():
    kd0_range = np.linspace(0.0, 100, 200).reshape(-1, 1)
    return kd0_range

def plot_gp_results_1d(kp0_values, kd0_values, tracking_errors, gp_kp0, gp_kd0):
    # Create prediction ranges
    kp0_pred = create_prediction_range_kp0()
    kd0_pred = create_prediction_range_kd0()
    
    # Predict for kp0
    y_mean_kp0, y_std_kp0 = gp_kp0.predict(kp0_pred, return_std=True)
    
    # Predict for kd0
    y_mean_kd0, y_std_kd0 = gp_kd0.predict(kd0_pred, return_std=True)
    

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(kp0_pred, y_mean_kp0, 'k-', lw=1.5, zorder=9, label='Mean prediction')
    plt.fill_between(kp0_pred.ravel(), y_mean_kp0 - 1.96 * y_std_kp0, y_mean_kp0 + 1.96 * y_std_kp0,
                    alpha=0.5, fc='orange', ec='None', label='95% confidence interval')
    #plt.scatter(X, y, c='blue', s=50, zorder=10, edgecolors=(0, 0, 0), label='Observations')
    #plt.plot(X_pred, np.sin(X_pred), 'r:', lw=1.5, label='True function')
    plt.title("Gaussian process regression on noise-free dataset")
    plt.xlabel('X')
    plt.ylabel('f(X)')
    plt.legend(loc='upper left')
    #plt.show()




    # Plot for kp0
    #plt.figure(figsize=(14, 6))
    
    #plt.subplot(1, 2, 1)
    #plt.fill_between(
    #    kp0_pred.ravel(),
    #    y_mean_kp0 - 1.96 * y_std_kp0,
    #    y_mean_kp0 + 1.96 * y_std_kp0,
    #    alpha=0.2,
    #    label='95% Confidence Interval'
    #)
    #plt.plot(kp0_pred, y_mean_kp0, 'k-', label='GP Mean Prediction')
    #plt.scatter(kp0_values, tracking_errors, c='r', label='Data Points')
    #plt.title('GP Regression for kp0')
    #plt.xlabel('kp0')
    #plt.ylabel('Tracking Error')
    #plt.legend()
    
    # Plot for kd0
    plt.subplot(1, 2, 2)
    plt.plot(kd0_pred, y_mean_kd0, 'k-', lw=1.5, zorder=9, label='Mean prediction')
    plt.fill_between(kd0_pred.ravel(), y_mean_kd0 - 1.96 * y_std_kd0, y_mean_kd0 + 1.96 * y_std_kd0,
                    alpha=0.5, fc='orange', ec='None', label='95% confidence interval')
    #plt.scatter(X, y, c='blue', s=50, zorder=10, edgecolors=(0, 0, 0), label='Observations')
    #plt.plot(X_pred, np.sin(X_pred), 'r:', lw=1.5, label='True function')
    plt.title("Gaussian process regression on noise-free dataset")
    plt.xlabel('X')
    plt.ylabel('f(X)')
    plt.legend(loc='upper left')



    #plt.fill_between(
    #    kd0_pred.ravel(),
    #    y_mean_kd0 - 1.96 * y_std_kd0,
    #    y_mean_kd0 + 1.96 * y_std_kd0,
    #    alpha=0.2,
    #    label='95% Confidence Interval'
    #)
    #plt.plot(kd0_pred, y_mean_kd0, 'k-', label='GP Mean Prediction')
    #plt.scatter(kd0_values, tracking_errors, c='r', label='Data Points')
    #plt.title('GP Regression for kd0')
    #plt.xlabel('kd0')
    #plt.ylabel('Tracking Error')
    #plt.legend()
    
    plt.tight_layout()
    plt.show()