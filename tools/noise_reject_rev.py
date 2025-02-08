import glob
import os
import random

import numpy as np
from scipy import signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import pandas as pd

from math import ceil
import sys
import argparse

from wave_wavfile_rev import Wave_info

'''
ノイズ処理を行うclass
基本的にstftをして、時間周波数方向で音声強調を行う
'''
class Noise(Wave_info):
	def __init__(self,path,sampling=None,
		n_fft=2048,hop_length=128,noise_path=None,
		alpha=5.0,mu=1e2, cof=1.0):
		super().__init__(path,sampling,n_fft,hop_length, cof)
		self.alpha=alpha
		self.mu=mu
		if noise_path!=None:
			self.noise_wave = Wave_info(noise_path,sampling=self.samplerate).wave
		else:
			self.noise_wave = self.wave[4096:12288]

	def spectrumSubtruction(self,average=True):
		p=1
		wave = self.wave.copy()

		'''
		for i in range(2):
			X,arg = self.stft(self.wave[:,i])
			input_power =np.power(X,p)
			noise = self.noise_wave[:,i]
			noise_abs, _ =self.stft(noise)
			if average==True:
				noise_power = np.mean(np.power(noise_abs,p),axis=1,keepdims=True) 
			else:
				noise_power = np.power(noise_abs,p)

			eps = 1e-10*input_power
			abs = np.power(np.maximum(input_power-self.alpha*noise_power,eps), 1./p)
			denoised = self.istft(abs,arg)
			remain = denoised.shape[0] - wave[:,0].shape[0]
			left = remain // 2
			right = remain - left
			print(denoised.shape[0], wave[:,0].shape[0], remain, left, right)
			wave[:,i] = self.istft(abs,arg)[left:wave[:,0].shape[0]+right]
		'''

		X,arg = self.stft(self.wave)
		input_power =np.power(X,p)
		noise = self.noise_wave
		noise_abs, _ =self.stft(noise)
		if average==True:
			tmp = np.power(noise_abs, p)
			self.noise_power = np.mean(tmp, axis=1, keepdims=True)
			#self.noise_power = np.mean(np.power(noise_abs,p),axis=1,keepdims=True)
		else:
			self.noise_power = np.power(noise_abs,p)

		eps = 1e-10*input_power
		abs = np.power(np.maximum(input_power-self.alpha*self.noise_power,eps), 1./p)
		denoised = self.istft(abs,arg)
		remain = denoised.shape[0] - wave.shape[0]
		left = remain // 2
		right = remain - left
		print(denoised.shape[0], wave.shape[0], remain, left, right)
		wave = self.istft(abs,arg)[left:wave.shape[0]+right]

		print(wave)
		return wave


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--ifile')
	parser.add_argument('-o', '--ofile')
	parser.add_argument('-a', '--alpha', type=float, default=4.0)
	parser.add_argument('-c', '--cof', type=float, default=1.0)
	parser.add_argument('--Fs', type=int, default=20000)
	parser.add_argument('--hop', type=int, default=2)
	parser.add_argument('--fftlen', type=int, default=1024)
	parser.add_argument('--nFiles', '-n', action='store_true')
	# parser.add_argument('-i', '--input',type=argparse.FileType('r'), default='-')
	parser.add_argument('--verbose', '-v', action='store_true')
	parser.add_argument('--debug', '-d', action='store_true')
	parser.add_argument('--log', default='')
	args = parser.parse_args()

	'''
	with open('./split_data_list.txt') as f:
		for line in f:
			line = line.rstrip()
			if line[-1]=='/':
				line = line[:-1]
			dir_path = (line + '/*.WAV')
			parent_dir = os.path.dirname(line)
			print(parent_dir)
			print(line)
			denoised_dir = parent_dir + '/WAV_denoised'
			if not os.path.isdir(denoised_dir):
				os.mkdir(denoised_dir)
			print(dir_path)
			fileList=glob.glob(dir_path)
			for i,path in enumerate(fileList):
				print(path)
				# noise = Noise(path,sampling=44100,hop_length=4,alpha=3.0,mu=1e2)
				noise = Noise(path,sampling=44100,hop_length=4,alpha=4.0,mu=1e2)
				data = noise.spectrumSubtruction()
				# noise.plotWave(noise.noise_wave)
				print(denoised_dir + '/{}'.format(os.path.basename(path)))
				wavfile.write(denoised_dir + '/{}'.format(os.path.basename(path)),noise.samplerate,data.astype(np.int16))
	'''
	#noise = Noise(sys.argv[1], sampling=44100, hop_length=4, alpha=4.0, mu=1e2)
	#noise_files = ['noise_data/noi_004.wav','noise_data/noi_005.wav','noise_data/noi_006.wav','noise_data/noi_007.wav','noise_data/noi_008.wav']

	'''
	if sys.argv[1] == 'nofile':
		noise = Noise(sys.argv[2], sampling=20000, n_fft=1024, hop_length=2, alpha=4.0, mu=1e2)
	else:
		noise = Noise(sys.argv[2], sampling=20000, n_fft=1024, hop_length=2, alpha=4.0, mu=1e2, noise_path=sys.argv[1])
	'''

	if args.nFiles:
		noise_files = ['noise_data/20220727_28/noi27_010.wav', 'noise_data/20220727_28/noi27_050.wav',
					   'noise_data/20220727_28/noi27_090.wav', 'noise_data/20220727_28/noi27_120.wav',
					   'noise_data/20220727_28/noi28_010.wav', 'noise_data/20220727_28/noi28_030.wav',
					   'noise_data/20220727_28/noi28_050.wav', ]
		nfn = random.choice(noise_files)
	else:
		nfn = None
	noise = Noise(args.ifile, sampling=args.Fs, n_fft=args.fftlen, hop_length=args.hop, alpha=args.alpha, mu=1e2, noise_path=nfn, cof=args.cof)
	# noise = Noise(sys.argv[1], sampling=20000, n_fft=1024, hop_length=2, alpha=3.0, mu=1e2, noise_path=None)
	data = noise.spectrumSubtruction()
	wavfile.write(args.ofile, noise.samplerate, data.astype(np.int16))
