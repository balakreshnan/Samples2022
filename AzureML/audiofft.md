# Azure ML Processing Audio Spectrum  analysis

## Use Azure ML to process audio spectrum analysis

## pre-requisites

- Azure Account
- Azure Machine Learning Workspace
- Azure Storage
- Sample audio file
- Audio file used is sample i got from internet
- the plots are only samples

## Code

- Read audio file

```
from scipy.io import wavfile # scipy library to read wav files
import numpy as np

AudioName = "input.wav" # Audio File
fs, Audiodata = wavfile.read(AudioName)
```

- plot the output

```
# Plot the audio signal in time
import matplotlib.pyplot as plt
plt.plot(Audiodata)
plt.title('Audio signal in time',size=16)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/fft1.jpg "Architecture")

- now spectrum analysis

```
# spectrum
from scipy.fftpack import fft # fourier transform
n = len(Audiodata) 
AudioFreq = fft(Audiodata)
AudioFreq = AudioFreq[0:int(np.ceil((n+1)/2.0))] #Half of the spectrum
MagFreq = np.abs(AudioFreq) # Magnitude
MagFreq = MagFreq / float(n)
# power spectrum
MagFreq = MagFreq**2
if n % 2 > 0: # ffte odd 
    MagFreq[1:len(MagFreq)] = MagFreq[1:len(MagFreq)] * 2
else:# fft even
    MagFreq[1:len(MagFreq) -1] = MagFreq[1:len(MagFreq) - 1] * 2 
```

- plot the output

```
plt.figure()
freqAxis = np.arange(0,int(np.ceil((n+1)/2.0)), 1.0) * (fs / n);
plt.plot(freqAxis/1000.0, 10*np.log10(MagFreq)) #Power spectrum
plt.xlabel('Frequency (kHz)'); plt.ylabel('Power spectrum (dB)');
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/fft2.jpg "Architecture")

- now analyze frequency

```
from scipy.fft import fft, fftfreq

yf = fft(Audiodata)
xf = fftfreq(n, 1 / SAMPLE_RATE)

plt.plot(xf, np.abs(yf))
plt.show()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/fft3.jpg "Architecture")

- Inverse frequency

```
from scipy.fft import irfft

new_sig = irfft(yf)

plt.plot(new_sig[:1000])
plt.show()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/fft4.jpg "Architecture")

- Now frequency filter and spectrum analysis

```
mag_spectrum = np.abs(AudioFreq)
plt.figure(figsize=(18,5))
frequency = np.linspace(0, SAMPLE_RATE, len(mag_spectrum))
num_frequency_bins = int(len(frequency) * 0.1)
plt.plot(frequency[:num_frequency_bins], mag_spectrum[:num_frequency_bins])
plt.xlabel("Frequency (hz)")
plt.ylabel("Sawblade")
plt.show()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/fft5.jpg "Architecture")