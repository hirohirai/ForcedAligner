# ForcedAligner


- atr文の疑似ノイズ削除後のデータを作成
for spk in ../DBS_/atr/wav/F*;do;for fn in $spk/?/???.wav; do;fnb=${fn:t};echo $fn; python tools/noise_reject_rev.py --nFiles -c 0.2 -a 3 -i $fn -o ../DBS_/atr/nr_wav/${spk:t}_2/${fnb:0:1}/${fnb};done;done
 - F0を計算する。
  - for fn in ../../DBS_/rtmri-atr503/wav/s1/*.wav ;do; ./calF0.py --min_f0 40 --max_f0 420 -i $fn -o Result_rtmri/s1/${fn:r:t}.f0;done
- worldを計算する
  - for fn in *.wav;do;python ~/work_local/Speech/ForceAligner/tools/world.py $fn --f0 ~/work_local/Speech/DBS_/rtmri-atr503/F0/s31/${fn:r:t}.f0 --odir ../../world/s31 ;done

## 学習用データ作成
- preprocess.py
- makestats.py
- preprocess1.py
- 学習用リストを作成
```commandline
./tools/div.py -i data/dataset_input.csv --rate2 -1.0 --rate 0.9
```
- train.py

## RTMRIの処理
- denoise wavを作成
- for ifn in ~/Dropbox/realTimeMRI/20220727/s31/WAV_R/1001/split/*.WAV;do;  python tools/noise_reject_rev.py -i $ifn -o wav_denoise/${ifn:r:t}.wav;done
- 文単位に分ける（２日目b）
- python tools/mri_exel_to_list.py /home/hirai/Dropbox/realTimeMRI/ATR503/s33/20240205_sentence_pos.csv >| List/s33_a_al.lst --all -s 0
- python tools/split_file.py wav_denoise TextGrid_ref -l List/s33_b_al.lst --owav out/wav -o out/TextGrid --fnb b
- F0, World 20KHz 
- 予測
- for fn in out2/world/*.npz;do;python3 pred.py -i $fn --wcof data/stats/world.npz -w exp/best_loss.pth -m exp/model.yaml -o out2/npy/${fn:r:t}.npy;done
- TextGridへ変換
- for fn in out/npy/s31/*.npy;do; echo $fn; python3 tools/fa_dp.py --st_width -1.0 out/TextGrid_o/s31/${fn:r:t}.TextGrid $fn >| out/TextGrid/s31/${fn:r:t}.TextGrid --dur_weight 10.0;done

for fn in out3/TextGrid_fao/s25/*.TextGrid ;do; echo $fn; python tools/syuusei.py $fn /home/hirai/work_local/Speech/DBS_/rt-atr503/TextGrid/s25/${fn:r:t}.TextGrid;done
