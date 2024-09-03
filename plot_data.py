import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(1,3)
figure.set_figheight(4)
figure.set_figwidth(16)
figure.subplots_adjust(wspace=0.4)

#Iter_T = np.linspace(20,1000,20)
#Iter_T = np.linspace(500,20000,20)
##### Data at d=40, d0=4
Regret_MS_mean = []
Regret_MS_std = []
Regret_Lasso_mean = []
Regret_Lasso_std = []
##########################################################################################################################################################
starting_point = 0

excel_file = "data_greedy_40" + ".xlsx"  # Update with your Excel file name/path
df = pd.read_excel(excel_file)
Iter_T = np.linspace(200,20000,20)
for T in Iter_T[starting_point:]:
    # Model selection regret
    print(type(df['Regret_MS_'+ str(T)]))
    mean_error_MS = df['Regret_MS_'+ str(T)].mean()
    std_error_MS = df['Regret_MS_'+ str(T)].std()

    Regret_MS_mean.append(mean_error_MS)
    Regret_MS_std.append(std_error_MS)

excel_file = "data_Lasso_NC_40" + ".xlsx"  # Update with your Excel file name/path
df = pd.read_excel(excel_file)
Iter_T = np.linspace(200, 20000, 20)
for T in Iter_T[starting_point:]:
    # Lasso regret
    mean_error_Lasso = df['Regret_Lasso_' + str(T)].mean()
    std_error_Lasso = df['Regret_Lasso_' + str(T)].std()

    Regret_Lasso_mean.append(mean_error_Lasso)
    Regret_Lasso_std.append(std_error_Lasso)

Regret_MS_mean_array = np.array(Regret_MS_mean)
Regret_MS_std_array = np.array(Regret_MS_std)
Regret_Lasso_mean_array = np.array(Regret_Lasso_mean)
Regret_Lasso_std_array = np.array(Regret_Lasso_std)
# plot the log mean
#plt.figure(figsize=(6, 6))  # Adjust size if needed
axis[0].plot(Iter_T[starting_point:], Regret_MS_mean_array, 'blue', label='EMC - ours')
axis[0].fill_between(Iter_T[starting_point:], Regret_MS_mean_array - Regret_MS_std_array,Regret_MS_mean_array +  Regret_MS_std_array, color='skyblue', alpha=0.4)
#plt.plot(Iter_T[starting_point:], Regret_MS_mean_array, 'blue', label='EMTC - ours')
#plt.fill_between(Iter_T[starting_point:], Regret_MS_mean_array - Regret_MS_std_array,Regret_MS_mean_array +  Regret_MS_std_array, color='skyblue', alpha=0.4)

#plt.plot(Iter_T[starting_point:], Regret_Lasso_mean_array, 'red', label='ESTC - Lasso')
#plt.fill_between(Iter_T[starting_point:], Regret_Lasso_mean_array - Regret_Lasso_std_array,Regret_Lasso_mean_array +  Regret_Lasso_std_array, color='lightcoral', alpha=0.4)
axis[0].plot(Iter_T[starting_point:], Regret_Lasso_mean_array, 'red', label='ESTC - Lasso')
axis[0].fill_between(Iter_T[starting_point:], Regret_Lasso_mean_array - Regret_Lasso_std_array,Regret_Lasso_mean_array +  Regret_Lasso_std_array, color='lightcoral', alpha=0.4)

axis[0].set_xlabel('$T$ - number of rounds')  # Add label for x-axis
axis[0].set_ylabel('Cumulative regret')  # Add label for y-axis
axis[0].set_title('$d=40,\; d_0 = 4$')  # Add label for y-axis
axis[0].legend()  # Show legend
axis[0].grid(True)  # Add grid


##########################################################################################################################################################
Regret_MS_mean = []
Regret_MS_std = []
Regret_Lasso_mean = []
Regret_Lasso_std = []
starting_point = 0

excel_file = "data_greedy_80" + ".xlsx"  # Update with your Excel file name/path
df = pd.read_excel(excel_file)
Iter_T = np.linspace(200,20000,20)
Iter_T[starting_point] = 400
for T in Iter_T[starting_point:]:
    # Model selection regret
    print(type(df['Regret_MS_'+ str(T)]))
    mean_error_MS = df['Regret_MS_'+ str(T)].mean()
    std_error_MS = df['Regret_MS_'+ str(T)].std()

    Regret_MS_mean.append(mean_error_MS)
    Regret_MS_std.append(std_error_MS)

excel_file = "data_Lasso_NC_80" + ".xlsx"  # Update with your Excel file name/path
df = pd.read_excel(excel_file)
Iter_T = np.linspace(200,20000,20)
Iter_T[starting_point] = 400
for T in Iter_T[starting_point:]:
    # Lasso regret
    mean_error_Lasso = df['Regret_Lasso_' + str(T)].mean()
    std_error_Lasso = df['Regret_Lasso_' + str(T)].std()

    Regret_Lasso_mean.append(mean_error_Lasso)
    Regret_Lasso_std.append(std_error_Lasso)

Regret_MS_mean_array = np.array(Regret_MS_mean)
Regret_MS_std_array = np.array(Regret_MS_std)
Regret_Lasso_mean_array = np.array(Regret_Lasso_mean)
Regret_Lasso_std_array = np.array(Regret_Lasso_std)
# plot the log mean
#plt.figure(figsize=(6, 6))  # Adjust size if needed
axis[1].plot(Iter_T[starting_point:], Regret_MS_mean_array, 'blue', label='EMC - ours')
axis[1].fill_between(Iter_T[starting_point:], Regret_MS_mean_array - Regret_MS_std_array,Regret_MS_mean_array +  Regret_MS_std_array, color='skyblue', alpha=0.4)
#plt.plot(Iter_T[starting_point:], Regret_MS_mean_array, 'blue', label='EMTC - ours')
#plt.fill_between(Iter_T[starting_point:], Regret_MS_mean_array - Regret_MS_std_array,Regret_MS_mean_array +  Regret_MS_std_array, color='skyblue', alpha=0.4)

#plt.plot(Iter_T[starting_point:], Regret_Lasso_mean_array, 'red', label='ESTC - Lasso')
#plt.fill_between(Iter_T[starting_point:], Regret_Lasso_mean_array - Regret_Lasso_std_array,Regret_Lasso_mean_array +  Regret_Lasso_std_array, color='lightcoral', alpha=0.4)
axis[1].plot(Iter_T[starting_point:], Regret_Lasso_mean_array, 'red', label='ESTC - Lasso')
axis[1].fill_between(Iter_T[starting_point:], Regret_Lasso_mean_array - Regret_Lasso_std_array,Regret_Lasso_mean_array +  Regret_Lasso_std_array, color='lightcoral', alpha=0.4)

axis[1].set_xlabel('$T$ - number of rounds')  # Add label for x-axis
axis[1].set_ylabel('Cumulative regret')  # Add label for y-axis
axis[1].set_title('$d=80,\; d_0 = 10$')  # Add label for y-axis
axis[1].legend()  # Show legend
axis[1].grid(True)  # Add grid


##########################################################################################################################################################
Regret_MS_mean = []
Regret_MS_std = []
Regret_Lasso_mean = []
Regret_Lasso_std = []
starting_point = 0

excel_file = "data_greedy_100" + ".xlsx"  # Update with your Excel file name/path
df = pd.read_excel(excel_file)
Iter_T = np.linspace(500,20000,20)
for T in Iter_T[starting_point:]:
    # Model selection regret
    print(type(df['Regret_MS_'+ str(T)]))
    mean_error_MS = df['Regret_MS_'+ str(T)].mean()
    std_error_MS = df['Regret_MS_'+ str(T)].std()

    Regret_MS_mean.append(mean_error_MS)
    Regret_MS_std.append(std_error_MS)

excel_file = "data_Lasso_NC_100" + ".xlsx"  # Update with your Excel file name/path
df = pd.read_excel(excel_file)
Iter_T = np.linspace(500, 20000, 20)
for T in Iter_T[starting_point:]:
    # Lasso regret
    mean_error_Lasso = df['Regret_Lasso_' + str(T)].mean()
    std_error_Lasso = df['Regret_Lasso_' + str(T)].std()

    Regret_Lasso_mean.append(mean_error_Lasso)
    Regret_Lasso_std.append(std_error_Lasso)

Regret_MS_mean_array = np.array(Regret_MS_mean)
Regret_MS_std_array = np.array(Regret_MS_std)
Regret_Lasso_mean_array = np.array(Regret_Lasso_mean)
Regret_Lasso_std_array = np.array(Regret_Lasso_std)
# plot the log mean
#plt.figure(figsize=(6, 6))  # Adjust size if needed
axis[2].plot(Iter_T[starting_point:], Regret_MS_mean_array, 'blue', label='EMC - ours')
axis[2].fill_between(Iter_T[starting_point:], Regret_MS_mean_array - Regret_MS_std_array,Regret_MS_mean_array +  Regret_MS_std_array, color='skyblue', alpha=0.4)
#plt.plot(Iter_T[starting_point:], Regret_MS_mean_array, 'blue', label='EMTC - ours')
#plt.fill_between(Iter_T[starting_point:], Regret_MS_mean_array - Regret_MS_std_array,Regret_MS_mean_array +  Regret_MS_std_array, color='skyblue', alpha=0.4)

#plt.plot(Iter_T[starting_point:], Regret_Lasso_mean_array, 'red', label='ESTC - Lasso')
#plt.fill_between(Iter_T[starting_point:], Regret_Lasso_mean_array - Regret_Lasso_std_array,Regret_Lasso_mean_array +  Regret_Lasso_std_array, color='lightcoral', alpha=0.4)
axis[2].plot(Iter_T[starting_point:], Regret_Lasso_mean_array, 'red', label='ESTC - Lasso')
axis[2].fill_between(Iter_T[starting_point:], Regret_Lasso_mean_array - Regret_Lasso_std_array,Regret_Lasso_mean_array +  Regret_Lasso_std_array, color='lightcoral', alpha=0.4)

axis[2].set_xlabel('$T$ - number of rounds')  # Add label for x-axis
axis[2].set_ylabel('Cumulative regret')  # Add label for y-axis
axis[2].set_title('$d=100,\; d_0 = 15$')  # Add label for y-axis
axis[2].legend()  # Show legend
axis[2].grid(True)  # Add grid




#plt.ylabel('epochs')  # Add label for y-axis
#plt.title('Regret in case of interval partition ')  # Add title
#plt.legend()  # Show legend
#plt.grid(True)  # Add grid
#plt.savefig('IntervalPartitionRegret.png', dpi=300)
#plt.savefig('NonCrossingPartitionRegret100.png', dpi=300)

plt.savefig('NonCrossingPartitionRegret.png', dpi=300)
plt.show()

