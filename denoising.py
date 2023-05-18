import wfdb
import matplotlib.pyplot as plt
import warnings
import numpy as np
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
def WVT(signal,wavelet,level):
    signal=signal
    coeffs = pywt.wavedec(signal, wavelet, level)
    coeffs[1:] = (pywt.threshold(i,value=100,mode='soft')for i in coeffs[1:])
    denoised=pywt.waverec(coeffs, wavelet)
    return denoised

def ICA(signals,n,threshold):
    #ustawienie liczby komponentów ica
    ica = FastICA(n_components=n)
    #uzyskanie składowych niezależnych
    ica_sig = ica.fit_transform(signals.reshape(-1,n))
    #usunięcie szumu
    ica_sig[np.abs(ica_sig) < threshold] = 0
    #rekonstrukcja sygnału
    denoised=ica.inverse_transform(ica_sig)
    return denoised

def main():
    warnings.simplefilter("ignore")
    record_for_AR_WVT=wfdb.rdrecord('/home/pawel/physionet.org/files/pulse-transit-time-ppg/1.1.0/s3_run',channels=[1],sampto=4000)
    signal=record_for_AR_WVT.p_signal.flatten()
    record = wfdb.rdrecord('/home/pawel/physionet.org/files/pulse-transit-time-ppg/1.1.0/s3_run',channels=[1,2,3,6],sampto=4000)
    signals=record.p_signal
    #signals=np.transpose(signals)
    print(signals.shape)
    fs = record.fs
    t=1/fs
    t=len(signal)*t
    t = np.arange(0, t, 1/fs)

    wfdb.plot_wfdb(record, title='PPG')
    # Use the predict method to generate the denoised signal
    p=30

    denoised_signal_AR = AR(signal,p)
    denoised_signal_WVT = WVT(signal,'db4',4)
    denoised_signal_ICA=ICA(signals,4,0.006)

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
    plt.plot(t, signals, label='Noisy signal')
    plt.plot(t, denoised_signal_ICA[0:4000], label='Denoised signal_ICA')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('ICA')
    plt.legend()

    plt.show()
    plt.figure()
    plt.subplot(4, 1, 1)

    plt.plot(t, signals[:,0], label='Noisy signal')
    plt.plot(t, denoised_signal_ICA[:,0], label='Denoised signal_ICA')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('ICA\nch1')
    plt.legend()

    plt.subplot(4, 1, 2)

    plt.plot(t, signals[:,1], label='Noisy signal')
    plt.plot(t, denoised_signal_ICA[:, 1], label='Denoised signal_ICA')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('ch2')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(t, signals[:,2], label='Noisy signal')
    plt.plot(t, denoised_signal_ICA[:, 2], label='Denoised signal_ICA')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('ch3')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(t, signals[:,3], label='Noisy signal')
    plt.plot(t, denoised_signal_ICA[:, 3], label='Denoised signal_ICA')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('ch4')
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()