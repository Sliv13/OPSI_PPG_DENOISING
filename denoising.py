import wfdb
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
def main():
    warnings.simplefilter("ignore")

    record = wfdb.rdrecord('s1_walk',channels=[1],sampto=1000)
    signal=record.p_signal
    print(signal)
    fs = record.fs
    t=1/fs
    t=len(signal)*t
    t = np.arange(0, t, 1/fs)
    noise = 0.5 * np.random.randn(len(signal))
    signal_n=list()
    for element in signal:
        element=element+100 * np.random.randn()
        print(element)
        signal_n.append(element)
    signal=signal_n
    print(signal_n)
    # \signal = signal_n
    wfdb.plot_wfdb(record, title='PPG')
    p = 50
    model = AutoReg(signal, lags=p)

    # Fit the AR model to the signal
    results = model.fit()

    # Use the predict method to generate the denoised signal
    denoised_signal = results.predict(start=p, end=len(signal) - 1)

    # Plot the original noisy signal and the denoised signal
    plt.plot(t, signal, label='Noisy signal')
    plt.plot(t[p:], denoised_signal, label='Denoised signal')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()
    #wfdb.display(record.__dict__)
    print(record.__dict__)

if __name__=="__main__":
    main()