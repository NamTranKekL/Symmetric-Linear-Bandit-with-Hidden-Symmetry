import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import UnivariateSpline, CubicSpline

def moving_average(arr, window_size):
    # Compute the moving average only for valid elements
    moving_avg = np.convolve(arr, np.ones(window_size), 'valid') / window_size
    # Pad with the original elements for the beginning
    initial_elements = arr[:window_size - 1]
    return np.concatenate((initial_elements, moving_avg))

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(1,3)
figure.set_figheight(4)
figure.set_figwidth(16)
figure.subplots_adjust(wspace=0.4)

#Iter_T = np.linspace(20,1000,20)
#Iter_T = np.linspace(500,20000,20)
##### Data at d=40, d0=4
d = 80
d0 = 10
T_Start = 500
Regret_MS_mean = []
Regret_MS_std = []
Regret_Lasso_mean = []
Regret_Lasso_std = []
##########################################################################################################################################################
starting_point = 0

excel_file = "data_greedy_Sparsity_" + str(d) + ".xlsx"  # Update with your Excel file name/path
df = pd.read_excel(excel_file)
Iter_T = np.linspace(T_Start,20000,20)
for T in Iter_T[starting_point:]:
    # Model selection regret
    print(type(df['Regret_MS_'+ str(T)]))
    mean_error_MS = df['Regret_MS_'+ str(T)].mean()*math.sqrt(d)
    std_error_MS = df['Regret_MS_'+ str(T)].std()*math.sqrt(d)

    Regret_MS_mean.append(mean_error_MS)
    Regret_MS_std.append(std_error_MS)
Regret_MS_mean_array = np.array(Regret_MS_mean)
Regret_MS_std_array = np.array(Regret_MS_std)

spline_mean = UnivariateSpline(Iter_T, Regret_MS_mean_array, s=1200000)
axis[0].plot(Iter_T, spline_mean(Iter_T), 'blue', label='EMC - ours')
spline_std = UnivariateSpline(Iter_T, Regret_MS_std_array, s=1200000)
axis[0].fill_between(Iter_T, spline_mean(Iter_T) - spline_std(Iter_T),spline_mean(Iter_T) + spline_std(Iter_T), color='skyblue', alpha=0.4)

excel_file = "data_Lasso_Sparsity_" + str(d) + ".xlsx"  # Update with your Excel file name/path
df = pd.read_excel(excel_file)
Iter_T = np.linspace(T_Start, 20000, 100)
for T in Iter_T[starting_point:]:
    # Lasso regret
    mean_error_Lasso = df['Regret_Lasso_' + str(T)].mean()*math.sqrt(d)
    std_error_Lasso = df['Regret_Lasso_' + str(T)].std()*math.sqrt(d)

    Regret_Lasso_mean.append(mean_error_Lasso)
    Regret_Lasso_std.append(std_error_Lasso)


Regret_Lasso_mean_array = np.array(Regret_Lasso_mean)
Regret_Lasso_std_array = np.array(Regret_Lasso_std)
# plot the log mean
#plt.figure(figsize=(6, 6))  # Adjust size if needed
#plt.plot(Iter_T[starting_point:], Regret_MS_mean_array, 'blue', label='EMTC - ours')
#plt.fill_between(Iter_T[starting_point:], Regret_MS_mean_array - Regret_MS_std_array,Regret_MS_mean_array +  Regret_MS_std_array, color='skyblue', alpha=0.4)

#plt.plot(Iter_T[starting_point:], Regret_Lasso_mean_array, 'red', label='ESTC - Lasso')
#plt.fill_between(Iter_T[starting_point:], Regret_Lasso_mean_array - Regret_Lasso_std_array,Regret_Lasso_mean_array +  Regret_Lasso_std_array, color='lightcoral', alpha=0.4)
spline_mean = UnivariateSpline(Iter_T, Regret_Lasso_mean_array, s=120000000)
axis[0].plot(Iter_T, spline_mean(Iter_T), 'red', label='ESTC - Lasso')
spline_std = UnivariateSpline(Iter_T, Regret_Lasso_std_array, s=120000000)
axis[0].fill_between(Iter_T, spline_mean(Iter_T) - spline_std(Iter_T),spline_mean(Iter_T) + spline_std(Iter_T), color='lightcoral', alpha=0.4)

axis[0].set_xlabel('$T$ - number of rounds')  # Add label for x-axis
axis[0].set_ylabel('Cumulative regret')  # Add label for y-axis
axis[0].set_title('Sparsity, $d=80, \; d_0=10$')  # Add label for y-axis
axis[0].legend()  # Show legend
axis[0].grid(True)  # Add grid


##########################################################################################################################################################
Regret_MS_mean = []
Regret_MS_std = []
Regret_Lasso_mean = []
Regret_Lasso_std = []
starting_point = 0

excel_file = "data_greedy_NC_" + str(d) + ".xlsx"  # Update with your Excel file name/path
df = pd.read_excel(excel_file)
Iter_T = np.linspace(T_Start,20000,20)
for T in Iter_T[starting_point:]:
    # Model selection regret
    print(type(df['Regret_MS_'+ str(T)]))
    mean_error_MS = df['Regret_MS_'+ str(T)].mean()*math.sqrt(d)
    std_error_MS = df['Regret_MS_'+ str(T)].std()*math.sqrt(d)*2

    Regret_MS_mean.append(mean_error_MS)
    Regret_MS_std.append(std_error_MS)
Regret_MS_mean_array = np.array(Regret_MS_mean)
Regret_MS_std_array = np.array(Regret_MS_std)

spline_mean = UnivariateSpline(Iter_T, Regret_MS_mean_array, s=120000)
axis[1].plot(Iter_T, spline_mean(Iter_T), 'blue', label='EMC - ours')
spline_std = UnivariateSpline(Iter_T, Regret_MS_std_array, s=120000)
axis[1].fill_between(Iter_T, spline_mean(Iter_T) - spline_std(Iter_T),spline_mean(Iter_T) + spline_std(Iter_T), color='skyblue', alpha=0.4)



excel_file = "data_Lasso_NC_"+ str(d) + ".xlsx"  # Update with your Excel file name/path
df = pd.read_excel(excel_file)
Iter_T = np.linspace(T_Start, 20000, 100)
for T in Iter_T[starting_point:]:
    # Lasso regret
    mean_error_Lasso = df['Regret_Lasso_' + str(T)].mean()*math.sqrt(d)
    std_error_Lasso = df['Regret_Lasso_' + str(T)].std()*math.sqrt(d)

    Regret_Lasso_mean.append(mean_error_Lasso)
    Regret_Lasso_std.append(std_error_Lasso)


Regret_Lasso_mean_array = np.array(Regret_Lasso_mean)
Regret_Lasso_std_array = np.array(Regret_Lasso_std)
# plot the log mean
#plt.figure(figsize=(6, 6))  # Adjust size if needed
#plt.plot(Iter_T[starting_point:], Regret_MS_mean_array, 'blue', label='EMTC - ours')
#plt.fill_between(Iter_T[starting_point:], Regret_MS_mean_array - Regret_MS_std_array,Regret_MS_mean_array +  Regret_MS_std_array, color='skyblue', alpha=0.4)

#plt.plot(Iter_T[starting_point:], Regret_Lasso_mean_array, 'red', label='ESTC - Lasso')
#plt.fill_between(Iter_T[starting_point:], Regret_Lasso_mean_array - Regret_Lasso_std_array,Regret_Lasso_mean_array +  Regret_Lasso_std_array, color='lightcoral', alpha=0.4)
spline_mean = UnivariateSpline(Iter_T, Regret_Lasso_mean_array, s=120000000)
axis[1].plot(Iter_T, spline_mean(Iter_T), 'red', label='ESTC - Lasso')
spline_std = UnivariateSpline(Iter_T, Regret_Lasso_std_array, s=120000000)
axis[1].fill_between(Iter_T, spline_mean(Iter_T) - spline_std(Iter_T),spline_mean(Iter_T) + spline_std(Iter_T), color='lightcoral', alpha=0.4)

axis[1].set_xlabel('$T$ - number of rounds')  # Add label for x-axis
axis[1].set_ylabel('Cumulative regret')  # Add label for y-axis
axis[1].set_title('Non-crossing Partition, $d=80, \; d_0=10$')  # Add label for y-axis
axis[1].legend()  # Show legend
axis[1].grid(True)  # Add grid


##########################################################################################################################################################
Regret_MS_mean = []
Regret_MS_std = []
Regret_Lasso_mean = []
Regret_Lasso_std = []

starting_point = 0

excel_file = "data_greedy_NN_"+ str(d) + ".xlsx"  # Update with your Excel file name/path
df = pd.read_excel(excel_file)
Iter_T = np.linspace(T_Start,20000,20)
for T in Iter_T[starting_point:]:
    # Model selection regret
    print(type(df['Regret_MS_'+ str(T)]))
    mean_error_MS = df['Regret_MS_'+ str(T)].mean()*math.sqrt(d)
    std_error_MS = df['Regret_MS_'+ str(T)].std()*math.sqrt(d)*2

    Regret_MS_mean.append(mean_error_MS)
    Regret_MS_std.append(std_error_MS)
Regret_MS_mean_array = np.array(Regret_MS_mean)
Regret_MS_std_array = np.array(Regret_MS_std)

spline_mean = UnivariateSpline(Iter_T, Regret_MS_mean_array, s=120000)
axis[2].plot(Iter_T, spline_mean(Iter_T), 'blue', label='EMC - ours')
spline_std = UnivariateSpline(Iter_T, Regret_MS_std_array, s=120000)
axis[2].fill_between(Iter_T, spline_mean(Iter_T) - spline_std(Iter_T),spline_mean(Iter_T) + spline_std(Iter_T), color='skyblue', alpha=0.4)


excel_file = "data_Lasso_NN_" + str(d) + ".xlsx"  # Update with your Excel file name/path
df = pd.read_excel(excel_file)
Iter_T = np.linspace(T_Start, 20000, 100)
for T in Iter_T[starting_point:]:
    # Lasso regret
    mean_error_Lasso = df['Regret_Lasso_' + str(T)].mean()*math.sqrt(d)
    std_error_Lasso = df['Regret_Lasso_' + str(T)].std()*math.sqrt(d)

    Regret_Lasso_mean.append(mean_error_Lasso)
    Regret_Lasso_std.append(std_error_Lasso)


Regret_Lasso_mean_array = np.array(Regret_Lasso_mean)
Regret_Lasso_std_array = np.array(Regret_Lasso_std)
# plot the log mean
#plt.figure(figsize=(6, 6))  # Adjust size if needed
#plt.plot(Iter_T[starting_point:], Regret_MS_mean_array, 'blue', label='EMTC - ours')
#plt.fill_between(Iter_T[starting_point:], Regret_MS_mean_array - Regret_MS_std_array,Regret_MS_mean_array +  Regret_MS_std_array, color='skyblue', alpha=0.4)

#plt.plot(Iter_T[starting_point:], Regret_Lasso_mean_array, 'red', label='ESTC - Lasso')
#plt.fill_between(Iter_T[starting_point:], Regret_Lasso_mean_array - Regret_Lasso_std_array,Regret_Lasso_mean_array +  Regret_Lasso_std_array, color='lightcoral', alpha=0.4)
spline_mean = UnivariateSpline(Iter_T, Regret_Lasso_mean_array, s=120000000)
axis[2].plot(Iter_T, spline_mean(Iter_T), 'red', label='ESTC - Lasso')
spline_std = UnivariateSpline(Iter_T, Regret_Lasso_std_array, s=120000000)
axis[2].fill_between(Iter_T, spline_mean(Iter_T) - spline_std(Iter_T),spline_mean(Iter_T) + spline_std(Iter_T), color='lightcoral', alpha=0.4)

axis[2].set_xlabel('$T$ - number of rounds')  # Add label for x-axis
axis[2].set_ylabel('Cumulative regret')  # Add label for y-axis
axis[2].set_title('Non-nesting Partition, $d=80, \; d_0=10$')  # Add label for y-axis
axis[2].legend()  # Show legend
axis[2].grid(True)  # Add grid



#plt.ylabel('epochs')  # Add label for y-axis
#plt.title('Regret in case of interval partition ')  # Add title
#plt.legend()  # Show legend
#plt.grid(True)  # Add grid
#plt.savefig('IntervalPartitionRegret.png', dpi=300)
#plt.savefig('NonCrossingPartitionRegret100.png', dpi=300)

plt.savefig('EMCRegretd80.png', dpi=300)
plt.show()

