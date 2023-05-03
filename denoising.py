import wfdb
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import pywt
from sklearn.decomposition import FastICA
def add_noise(signal):
    signal_n=list()
    for element in signal:
        element=element+100 * np.random.randn()
        print(element)
        signal_n.append(element)
    return signal_n
def AR(signal,p):
    model = AutoReg(signal, lags=p)
    # Fit the AR model to the signal
    results = model.fit()
    # Use the predict method to generate the denoised signal
    return results.predict(start=p, end=len(signal) - 1)
def WVT(signal,wavelet,level,threshold):
    coeffs = pywt.wavedec(signal, wavelet, level)

    def soft_thresh(x, t):
        return np.sign(x) * np.maximum(np.abs(x) - t, 0.)

    threshold = threshold
    coeffs[1:] = [soft_thresh(i, threshold) for i in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def ICA(signal,n):
    ica = FastICA(n_components=n)
    ica_sig = ica.fit_transform(signal.reshape(-1, 1))
    denoised=ica.inverse_transform(ica_sig)
    return denoised

def main():
    warnings.simplefilter("ignore")

    record = wfdb.rdrecord('wrist-ppg-during-exercise-1.0.0/s1_walk',channels=[1],sampto=1000)
    signal=record.p_signal
    print(signal)
    fs = record.fs
    t=1/fs
    t=len(signal)*t
    t = np.arange(0, t, 1/fs)

    signal = add_noise(signal)
    signal=np.array(signal)
    # \signal = signal_n
    wfdb.plot_wfdb(record, title='PPG')
    # Use the predict method to generate the denoised signal
    p=5

    denoised_signal_AR = AR(signal,p)
    denoised_signal_WVT = WVT(signal,'db1',5,100)
    denoised_signal_ICA=ICA(signal,16)

    # Plot the original noisy signal and the denoised signal
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t, signal, label='Noisy signal')
    plt.plot(t[p:], denoised_signal_AR, label='Denoised signal_AR')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Model AR')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(t, signal, label='Noisy signal')
    plt.plot(t, denoised_signal_WVT, label='Denoised signal_WVT')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Transformata Falkowa')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, signal, label='Noisy signal')
    plt.plot(t, denoised_signal_ICA, label='Denoised signal_WVT')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Transformata Falkowa')
    plt.legend()

    plt.show()
    #wfdb.display(record.__dict__)
    print(record.__dict__)

if __name__=="__main__":
    main()