import os
import glob
import librosa
'''
import soundfile
'''
import librosa.display
import numpy as np
import scipy.io.wavfile as wavfile
from fractions import Fraction
from scipy import signal
from matplotlib import pyplot as plt

class Wave_info():
	def __init__(self,path,sampling = None,n_fft=512,hop_length=256):
		'''
		pathからwavデータを読み取る
		samplingにサンプリング周期(主に16000)を入れるとダウンサンプリングする
		'''
		if isinstance(path, str):
			self.path = path
			self.samplerate,self.wave = wavfile.read(path)
		else:
			self.path = None
			self.wave = path
			self.samplerate = sampling

		if len(self.wave.shape) >1:
			self.wave = self.wave[:,1]
		
		# if sampling != None:
		#	self.wave = self.resampling(self.wave,sampling)
		#	self.samplerate = sampling

		self.len = len(self.wave)
		self.n_fft = n_fft
		self.hop_length =hop_length
		self.n_overlap = self.n_fft -self.hop_length

	def plotWave(self, wave, name = None):
		'''
		waveの形を描画する
		'''
		x = np.arange(0, self.len/self.samplerate, 1.0/self.samplerate) 
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(wave[0])
		if name ==None:
			name = os.path.splitext(os.path.basename(self.path))[0]
		plt.savefig('{}.png'.format(name))

	def plotSpecgram(self, wave, title=None, name =None, start=None,end=None):
		'''
		waveをスペクトログラムに直してからプロットする
		時間のデータ数はlen(wave) / (NFFT - noverlap) になる
		'''
		window = signal.windows.hann(self.n_fft)
		pxx, freqs, bins, im = plt.specgram(wave, NFFT=self.n_fft,
											Fs = self.samplerate, noverlap=self.n_overlap, 
											window=window,mode='magnitude')
		
		plt.axis([0, len(wave) / self.samplerate, 0, 8000])
		plt.xlabel("time [second]")
		plt.ylabel("frequency [Hz]")
		#plt.title(title+' spectral envelope')
		if name ==None:
			plt.savefig('plot.png')
		else:
			plt.savefig(name + '.png')
		plt.close('all')

	def plotMelSpecgram(self, wave, name ='plot', start=None,end=None):
		if wave.dtype != np.float:
			wave = wave.astype(np.float)
		 
		#wave = np.concatenate([wave,zero])
		S = librosa.feature.melspectrogram(y=wave, sr=self.samplerate,
										   n_fft=self.n_fft,hop_length=self.hop_length,n_mels=80,fmax=8000)

		S_dB = librosa.power_to_db(S, ref=np.max)
	   
		librosa.display.specshow(S_dB, sr=self.samplerate, 
								 hop_length=hop_length,
								 x_axis='time', y_axis='mel', fmax=8000, cmap='viridis'
								 )
		
		#plt.xlabel("time [second]")
		#plt.ylabel("frequency [Hz]")
		plt.title('melspectral envelope')
		plt.tight_layout()
		if name !=None:
			plt.savefig(name + '.png')
		plt.close('all')

	def resampling(self,wave,sample):
		'''
		waveのサンプリングレートを変更する
		'''
		a = Fraction(self.samplerate, sample)
		inSamplerate = a.numerator
		outSamplerate = a.denominator
		wave_downsample = signal.resample_poly(wave,outSamplerate, inSamplerate)
		wave_downsample = wave_downsample.astype(np.int16)
		return wave_downsample

	def stft(self,wave):
		_,_,stft = signal.stft(wave,fs=self.samplerate,
						   nperseg=self.n_fft,noverlap=self.n_overlap)
		sp_abs=  np.abs(stft)
		sp_arg = np.angle(stft)
		return sp_abs,sp_arg

	def istft(self,magnitude,arg=[],file_name=None):
		if arg==[]:
			wave = librosa.griffinlim(magnitude)
		else:
			process_stft = magnitude *(np.cos(arg) + np.sin(arg) * 1j)
			wave = signal.istft(process_stft,fs=self.samplerate,
								nperseg=self.n_fft,noverlap=self.n_overlap)
		return np.mean(wave,axis=0).astype(np.int16)
	
if __name__ == "__main__":
	path = ('./M/1.wav')
	wave = Wave_info(path,sampling=44100)
	wave_noise = wave.wave[2048:16384]
	wave.plotWave(wave_noise)
	wavfile.write(r'./raw.wav',wave.samplerate,wave_noise)