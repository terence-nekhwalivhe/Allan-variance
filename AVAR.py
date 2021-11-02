import numpy as np
import pandas as pd
from scipy.optimize import nnls
import matplotlib.pyplot as plt  # plotting
import seaborn as sns
import scipy.linalg as la 
import csv   

#Function to Open the CSV file
def get_rawdata():
    with open('12hr_100hz_RT.csv', 'r') as csv_file:
        csv_fil = csv.reader(csv_file)                       
        AXraw = []
        AYraw = []
        AZraw = []
        GXraw = []
        GYraw = []
        GZraw = []
        Temp = []
        Count = []
        Acc_d = []                            
        for row in csv_fil:
             AXraw.append(row[0])
             AYraw.append(row[1])
             AZraw.append(row[2])
             GXraw.append(row[3])
             GYraw.append(row[4])
             GZraw.append(row[5])
             Temp.append(row[6])
             Count.append(row[7])
#converting list of strings to int
        for i in range(1, len(Count)):
            AXraw[i] = float(AXraw[i])
            AYraw[i] = float(AYraw[i])
            AZraw[i] = float(AZraw[i])
            GXraw[i] = float(GXraw[i])
            GYraw[i] = float(GYraw[i])
            GZraw[i] = float(GZraw[i])
            Temp[i] = float(Temp[i])
            Count[i] = float(Count[i])
        acc_d = [AXraw, AYraw, AZraw, GXraw, GYraw, GZraw, Temp, Count]

 #Convert Python Nested Lists to Multidimensional NumPy Arrays
        dataArr = np.array(acc_d)
    
    return dataArr

'''Allan Deviation Computation
We will be computing the Allan deviation of gyro angle data in degrees. 
First, I imported the gyro rate data into numpy arrays. 
Then, I converted the raw gyro rate data from  to . 
Next, I Euler integrated the rate data to compute gyro orientation data in  
by computing the cumulative sum (numpy.cumsum()) of the data and multiplying it by the sample period (0.01sec). 
I then passed the gyro angle data to my Allan deviation function and computed the Allan variance in . 
Finally, I plotted the data on a log-scale plot.'''

fs = 100 # sample rate frequency
MEAS_DUR_SEC = 4320  #(12 Hours) seconds to record data
max_num_m = 1000 # number of clusters
n_samples = 12*60*60*fs # number of samples (12hours of 100Hz samples)
t_s = 1/fs # sample time
#theta = np.zeros(n_samples, dtype=np.float64) # ideal samples

'''def _compute_cluster_sizes(n_samples, t_s, tau_min, tau_max, n_clusters):
    if tau_min is None:
        min_size = 1
    else:
        min_size = int(tau_min / t_s)

    if tau_max is None:
        max_size = n_samples // 10
    else:
        max_size = int(tau_max / t_s)

    result = np.logspace(np.log2(min_size), np.log2(max_size),
                         num=n_clusters, base=2)

    return np.unique(np.round(result)).astype(int)'''

def AllanDeviation(dataArr, fs, maxNumM=100):
    """Compute the Allan deviation (sigma) of time-series data.
    ----
        dataArr (numpy.ndarray): 1D data array
        fs (int, float): Data sample frequency in Hz
        maxNumM (int): Number of output points
    
    Returns
    -------
        (taus, allanDev): Tuple of results
        taus (numpy.ndarray): Array of tau values
        allanDev (numpy.ndarray): Array of computed Allan deviations
    """
    ts = 1.0 / fs
    N = len(dataArr) # number of samples
    Mmax = 2**np.floor(np.log2(N / 2)) # maximum cluster size
    M = np.logspace(np.log10(1), np.log10(Mmax), num=maxNumM) # cluster sizes
    M = np.ceil(M).astype(int) # m must be an integer.
    M = np.unique(M)  # Remove duplicates
    taus = M * ts  # Compute 'cluster time' tau

    # Compute Allan variance
    allanVar = np.zeros(len(M))
    for i, mi in enumerate(M):
        twoMi = int(2 * mi)
        mi = int(mi)
        allanVar[i] = np.sum(
            (dataArr[twoMi:N] - (2.0 * dataArr[mi:N-mi]) + dataArr[0:N-twoMi])**2
        )
    
    allanVar /= (2.0 * taus**2) * (N - (2.0 * M))
    return (taus, np.sqrt(allanVar))  # Return deviation (dev = sqrt(var))

# Load CSV into np array
dataArr = get_rawdata()
fs = 100 # sample rate frequency
ts = 1.0 / fs

# Separate into arrays
gx = dataArr[3,:] * (180.0 / np.pi)  # [deg/s]
gy = dataArr[4,:] * (180.0 / np.pi)
gz = dataArr[5,:] * (180.0 / np.pi)

# Calculate gyro angles
thetax = np.cumsum(gx) * ts  # [deg]
thetay = np.cumsum(gy) * ts
thetaz = np.cumsum(gz) * ts

# Compute Allan deviations
(taux, adx) = AllanDeviation(thetax, fs, maxNumM=1000)
(tauy, ady) = AllanDeviation(thetay, fs, maxNumM=1000)
(tauz, adz) = AllanDeviation(thetaz, fs, maxNumM=1000)

# Plot data on log-scale
plt.figure()
plt.title('Gyro Allan Deviations')
plt.plot(taux, adx, label='gx')
plt.plot(tauy, ady, label='gy')
plt.plot(tauz, adz, label='gz')
plt.xlabel(r'$\tau$ [sec]')
plt.ylabel('Deviation [deg/sec]')
plt.grid(True, which="both", ls="-", color='0.65')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()

# find y intersection given a particular slope
def get_y_intersect(slope, tau, adev):
    logtau = np.log10(tau)
    logadev = np.log10(adev)
    dlogadev = np.diff(logadev) / np.diff(logtau)
    i = np.argmin(np.abs(dlogadev - slope))

    return logadev[i] - slope*logtau[i], i

# bias (in)stability B
slope = 0
b, i = get_y_intersect(slope, taux, adx)
scfB = np.sqrt(2*np.log(2)/np.pi)
logB = b - np.log10(scfB)
B=10**logB
line_B = B * scfB * np.ones(shape=(len(taux),1))
print("Bias instability:\n",B)

# rate random walk R
slope = 0.5
b, _ = get_y_intersect(slope, taux, adx)
logK = slope*np.log10(3) + b
K = 10**logK
line_K = K *np.sqrt(taux/3)
print("rate random walk:\n",K)

# ARW\VRW N
slope = -0.5
b, _ = get_y_intersect(slope, taux, adx)
logN = slope*np.log10(1) + b
N = 10**logN
line_N = N/(np.sqrt(taux))
print("ARW\RRW:\n",N)

#Rate ramp noise
slope = 1
b, _ = get_y_intersect(slope, taux, adx)
logR = slope*np.log10(np.sqrt(2)) + b
R = 10**logR
line_R= R*(taux/np.sqrt(2))
print("Rate ramp rate:\n",R)

#Quantization noise
slope = -1
b, _ = get_y_intersect(slope, taux, adx)
logT = slope*np.log10(np.sqrt(3)) + b
T = 10**logT
line_T = np.sqrt(3)*T/taux
print("Quantization:\n",T)

x_min = (10**(-2.5))
x_max = (10**(3.5))
y_min= (10**(0.5))
y_max=(10**(3))
# Plot data on log-scale
plt.figure()
plt.title('Gyro error analysis')
plt.plot(taux, adx, label='gx')
plt.plot(taux, line_B, label='Bias stability',  linestyle='dotted')
plt.plot(taux, line_K, label='Rate random walk', linestyle='dashed')
plt.plot(taux, line_N, label='ARW/VRW', linestyle='dashdot')
plt.plot(taux, line_R, label='Rate ramp')
plt.plot(taux, line_T, label='Quantization',linestyle='dotted' )
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xlabel(r'$\tau$ [sec]')
plt.ylabel('Deviation [deg/sec]')
plt.grid(True, which="both", ls="-", color='0.65')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()

'''def _compute_cluster_sizes(n_samples, dt, tau_min, tau_max, n_clusters):
    if tau_min is None:
        min_size = 1
    else:
        min_size = int(tau_min / dt)

    if tau_max is None:
        max_size = n_samples // 10
    else:
        max_size = int(tau_max / dt)

    result = np.logspace(np.log2(min_size), np.log2(max_size),
                         num=n_clusters, base=2)

    return np.unique(np.round(result)).astype(int)


def calculate_avar(theta, t0, max_num_m):
    n = theta.size # number of samples
    max_m = 2**int(np.log2(n/2)) # maximum cluster size
    m = np.logspace(np.log10(1), np.log10(max_m), max_num_m) # cluster sizes
    m = np.ceil(m).astype(int) # m must be an integer.
    m = np.unique(m) # Remove duplicates.

    tau = m*t0
    result = np.empty_like(m)
    for i in range(m.size):
        result[i] = np.sum((theta[2*m[i]:n] - 2*theta[m[i]:n-m[i]] + theta[:n-2*m[i]])**2)
    result = result / (2*tau**2 * (n - 2*m))

    return tau, result
# find y intersection given a particular slope
def get_y_intersect(slope, tau, adev):
    logtau = np.log10(tau)
    logadev = np.log10(adev)
    dlogadev = np.diff(logadev) / np.diff(logtau)
    i = np.argmin(np.abs(dlogadev - slope))

    return logadev[i] - slope*logtau[i], i

# noise density N
slope = -0.5
b, _ = get_y_intersect(slope, tau, adev)
logN = slope*np.log10(1) + b
N = 10**logN

# rate random walk R
slope = 0.5
b, _ = get_y_intersect(slope, tau, adev)
logK = slope*np.log10(3) + b
K = 10**logK

# bias (in)stability B
slope = 0
b, i = get_y_intersect(slope, tau, adev)
scfB = np.sqrt(2*np.log(2)/np.pi)
logB = b - np.log10(scfB)'''