"""
Plot H1 and L2 losses
Credits: Massimiliano Ghiotto
"""
import matplotlib.pyplot as plt
import numpy as np

import sys

params = {'legend.fontsize': 12,
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'xtick.labelsize': 11,
              'ytick.labelsize': 11}

plt.rcParams.update(params)

if len(sys.argv) < 2:
    n_plot = 0
else:
    n_plot = int(sys.argv[1])

match n_plot:
    case 0:
        #### Figure relative errors vs network width, fourier modes, and r_theta width
        # Example data
        network_width = np.array([400, 500, 600, 700, 800, 900, 1000])
        train_error_width = np.array([0.0149, 0.0198, 0.0165, 0.0129, 0.0136, 0.0155, 0.0210])
        test_error_width = np.array([0.0274, 0.0274, 0.0243, 0.0220, 0.0231, 0.0231, 0.0235])
        test_H1_error_width = np.array([0.1715, 0.1738, 0.1623, 0.1564, 0.1646, 0.1545, 0.1771])
        train_error_width_err = np.array([0.0034, 0.0004, 0.0016, 0.0025, 0.0013, 0.0019, 0.0030])
        test_error_width_err = np.array([0.0006, 0.0007, 0.0014, 0.0010, 0.0004, 0.0002, 0.005])
        test_H1_error_width_err = np.array([0.0164, 0.0024, 0.0064, 0.0075, 0.0092, 0.0018, 0.0102])

        fourier_modes = np.array([8, 16, 32, 64, 128])
        train_error_modes = np.array([0.0175, 0.0150, 0.0120, 0.0098, 0.0100])
        test_error_modes = np.array([0.0180, 0.0143, 0.0149, 0.0235, 0.0211])
        test_H1_error_modes = np.array([0.1614, 0.1249, 0.1082, 0.1412, 0.1591])
        train_error_modes_err = np.array([0.0053, 0.0041, 0.0021, 0.0027, 0.0017])
        test_error_modes_err = np.array([0.0018, 0.0023, 0.0039, 0.0075, 0.0023])
        test_H1_error_modes_err = np.array([0.0579, 0.0114, 0.0126, 0.0224, 0.0104])

        r_theta = np.array([8, 16, 32, 64, 128, 256])
        train_error_r_theta = np.array([0.0363, 0.0123, 0.0100, 0.0091, 0.0106, 0.0111])
        test_error_r_theta = np.array([0.0505, 0.0413, 0.0403, 0.0330, 0.0345, 0.0345])
        test_H1_error_r_theta = np.array([1.6399, 0.5657, 0.2850, 0.2640, 0.2309, 0.2627])
        train_error_r_theta_err = np.array([0.0025, 0.0010, 0.0003, 0.0018, 0.0006, 0.0008])
        test_error_r_theta_err = np.array([0.0080, 0.0040, 0.0018, 0.0005, 0.0010, 0.0018])
        test_H1_error_r_theta_err = np.array([0.2701, 0.0973, 0.0090, 0.0212, 0.0159, 0.0510])

        y_min = min(np.min(train_error_width), np.min(test_error_width), np.min(test_H1_error_width),
                    np.min(train_error_modes), np.min(test_error_modes), np.min(test_H1_error_modes), 
                    np.min(train_error_r_theta), np.min(test_error_r_theta), np.min(test_H1_error_r_theta))
        y_max = max(np.max(train_error_width), np.max(test_error_width), np.max(test_H1_error_width),
                    np.max(train_error_modes), np.max(test_error_modes), np.max(test_H1_error_modes),
                    np.max(train_error_r_theta), np.max(test_error_r_theta), np.max(test_H1_error_r_theta)) + 1
        err_max = max(np.max(train_error_width_err), np.max(train_error_modes_err), np.max(train_error_r_theta_err)) + 0.001

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # (a) L^2 Relative error vs Network Width
        axs[0].errorbar(network_width, train_error_width, yerr=train_error_width_err, fmt='-o', label=r'Train $L^{2}$ error', capsize=3)
        axs[0].errorbar(network_width, test_error_width, yerr=test_error_width_err, fmt='-o', label=r'Test $L^{2}$ error', capsize=3)
        axs[0].errorbar(network_width, test_H1_error_width, yerr=test_H1_error_width_err, fmt='-o', label=r'Test $H^{1}$ error', capsize=3)
        axs[0].set_title('Relative errors vs Network width')
        axs[0].set_xlabel('Network width')
        axs[0].set_xticks(network_width)
        axs[0].set_ylabel('Relative error')
        axs[0].set_yscale('log')
        axs[0].set_ylim([max(y_min-err_max, 0.0), y_max+err_max])
        axs[0].grid()
        axs[0].legend()

        # (b) L^2 Relative error vs Fourier Modes
        axs[1].errorbar(fourier_modes, train_error_modes, yerr=train_error_modes_err, fmt='-o', label=r'Train $L^{2}$ error', capsize=3)
        axs[1].errorbar(fourier_modes, test_error_modes, yerr=test_error_modes_err, fmt='-o', label=r'Test $L^{2}$ error', capsize=3)
        axs[1].errorbar(fourier_modes, test_H1_error_modes, yerr=test_H1_error_modes_err, fmt='-o', label=r'Test $H^{1}$ error', capsize=3)
        axs[1].set_title('Relative errors vs Fourier modes')
        axs[1].set_xlabel('Fourier modes')
        axs[1].set_xticks(fourier_modes)
        axs[1].set_ylim([max(y_min-err_max, 0.0), y_max+err_max])
        axs[1].set_yscale('log')
        axs[1].grid()
        axs[1].legend()

        # (c) L^2 Relative Error vs R_theta Width
        axs[2].errorbar(r_theta, train_error_r_theta, yerr=train_error_r_theta_err, fmt='-o', label=r'Train $L^{2}$ error', capsize=3)
        axs[2].errorbar(r_theta, test_error_r_theta, yerr=test_error_r_theta_err, fmt='-o', label=r'Test $L^{2}$ error', capsize=3)
        axs[2].errorbar(r_theta, test_H1_error_r_theta, yerr=test_H1_error_r_theta_err, fmt='-o', label=r'Test $H^{1}$ error', capsize=3)
        axs[2].set_title(r'Relative Errors vs $R_{\theta_t}$ width')
        axs[2].set_xlabel(r'$R_{\theta_t}$ width')
        axs[2].set_xticks(r_theta)
        axs[2].set_ylim([max(y_min-err_max, 0.0), y_max+err_max])
        axs[2].set_yscale('log')
        axs[2].grid()
        axs[2].legend()

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.savefig('no_scatter.eps',format='eps')
        plt.show()
    
    case 1:
        #### Data efficiency
        # Example data
        number_data = np.array([1000, 2000, 4000, 8000])
        train_error_DON = np.array([0.0162, 0.0129, 0.0131, 0.0124])
        test_error_DON = np.array([0.0293, 0.0220, 0.0213, 0.0163])
        test_H1_error_DON = np.array([0.1970, 0.1564, 0.1430, 0.1223])
        train_error_DON_err = np.array([0.0073, 0.0025, 0.0023, 0.0007])
        test_error_DON_err = np.array([0.0030, 0.0010, 0.0022, 0.0027])
        test_H1_error_DON_err = np.array([0.0151, 0.0075, 0.0104, 0.0056])

        train_error_FNO = np.array([0.0146, 0.0150, 0.0128, 0.0080])
        test_error_FNO = np.array([0.0139, 0.0143, 0.0137, 0.0097])
        test_H1_error_FNO = np.array([0.1210, 0.1249, 0.1224, 0.0675])
        train_error_FNO_err = np.array([0.0013, 0.0041, 0.0042, 0.0009])
        test_error_FNO_err = np.array([0.0011, 0.0023, 0.0027, 0.0025])
        test_H1_error_FNO_err = np.array([0.0075, 0.0114, 0.0113, 0.0066])

        train_error_WNO = np.array([0.0101, 0.0091, 0.0078, 0.0078])
        test_error_WNO = np.array([0.0356, 0.0330, 0.0265, 0.0183])
        test_H1_error_WNO = np.array([0.2734, 0.2640, 0.2073, 0.2144])
        train_error_WNO_err = np.array([0.0016, 0.0018, 0.0007, 0.0013])
        test_error_WNO_err = np.array([0.0008, 0.0005, 0.0008, 0.0015])
        test_H1_error_WNO_err = np.array([0.0234, 0.0212, 0.0253, 0.0357])

        train_error_r_theta = np.array([0.0363, 0.0123, 0.0100, 0.0091, 0.0106, 0.0111])
        test_error_r_theta = np.array([0.0505, 0.0413, 0.0403, 0.0330, 0.0345, 0.0345])
        test_H1_error_r_theta = np.array([1.6399, 0.5657, 0.2850, 0.2640, 0.2309, 0.2627])
        train_error_r_theta_err = np.array([0.0025, 0.0010, 0.0003, 0.0018, 0.0006, 0.0008])
        test_error_r_theta_err = np.array([0.0080, 0.0040, 0.0018, 0.0005, 0.0010, 0.0018])
        test_H1_error_r_theta_err = np.array([0.2701, 0.0973, 0.0090, 0.0212, 0.0159, 0.0510])

        y_min = min(np.min(train_error_DON), np.min(test_error_DON), np.min(test_H1_error_DON),
                    np.min(train_error_FNO), np.min(test_error_FNO), np.min(test_H1_error_FNO),
                    np.min(train_error_WNO), np.min(test_error_WNO), np.min(test_H1_error_WNO))
        y_max = max(np.max(train_error_DON), np.max(test_error_DON), np.max(test_H1_error_DON),
                    np.max(train_error_FNO), np.max(test_error_FNO), np.max(test_H1_error_FNO),
                    np.max(train_error_WNO), np.max(test_error_WNO), np.max(test_H1_error_WNO)) + 0.2

        err_max = max(np.max(train_error_DON_err), np.max(train_error_FNO_err), np.max(train_error_WNO_err)) + 0.001

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # (a) L^2 Relative error vs Network Width
        axs[0].errorbar(number_data, train_error_DON, yerr=train_error_DON_err, fmt='-o', label='Train $L^{2}$ error', capsize=3)
        axs[0].errorbar(number_data, test_error_DON, yerr=test_error_DON_err, fmt='-o', label='Test $L^{2}$ error', capsize=3)
        axs[0].errorbar(number_data, test_H1_error_DON, yerr=test_H1_error_DON_err, fmt='-o', label='Test $H^{1}$ error', capsize=3)
        # axs[0].set_title('DON')
        axs[0].set_xlabel('Total number of data')
        axs[0].set_xticks(number_data)
        # axs[0].set_xscale('log')    
        axs[0].set_ylabel('Relative error')
        axs[0].set_ylim([max(y_min-err_max, 0.005), y_max+err_max])
        axs[0].set_yscale('log')
        axs[0].grid()
        axs[0].legend()

        # (b) L^2 Relative error vs Fourier Modes
        axs[1].errorbar(number_data, train_error_FNO, yerr=train_error_FNO_err, fmt='-o', label='Train $L^{2}$ error', capsize=3)
        axs[1].errorbar(number_data, test_error_FNO, yerr=test_error_FNO_err, fmt='-o', label='Test $L^{2}$ error', capsize=3)
        axs[1].errorbar(number_data, test_H1_error_FNO, yerr=test_H1_error_FNO_err, fmt='-o', label='Test $H^{1}$ error', capsize=3)
        # axs[1].set_title('FNO')
        axs[1].set_xlabel('Total number of data')
        axs[1].set_xticks(number_data)
        # axs[1].set_xscale('log')    
        axs[1].set_ylim([max(y_min-err_max, 0.005), y_max+err_max])
        axs[1].set_yscale('log')
        axs[1].grid()
        axs[1].legend()

        # (c) L^2 Relative Error vs R_theta Width
        axs[2].errorbar(number_data, train_error_WNO, yerr=train_error_WNO_err, fmt='-o', label='Train $L^{2}$ error', capsize=3)
        axs[2].errorbar(number_data, test_error_WNO, yerr=test_error_WNO_err, fmt='-o', label='Test $L^{2}$ error', capsize=3)
        axs[2].errorbar(number_data, test_H1_error_WNO, yerr=test_H1_error_WNO_err, fmt='-o', label='Test $H^{1}$ error', capsize=3)
        # axs[2].set_title('WNO')
        axs[2].set_xlabel('Total number of data')
        axs[2].set_xticks(number_data)
        # axs[2].set_xscale('log')    
        axs[2].set_ylim([max(y_min-err_max, 0.005), y_max+err_max])
        axs[2].set_yscale('log')
        axs[2].grid()
        axs[2].legend()

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.savefig('data_efficiency.eps',format='eps')
        plt.show()