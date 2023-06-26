# ForcedAligner

- F0を計算する。
  - for fn in ../../DBS_/rtmri-atr503/wav/s1/*.wav ;do; ./calF0.py --min_f0 40 --max_f0 420 -i $fn -o Result_rtmri/s1/${fn:r:t}.f0;done
- worldを計算する
  - for fn in *.wav;do;python ~/work_local/Speech/auto_seg/tools/world.py $fn --f0 ~/work_local/Speech/DBS_/rtmri-atr503/F0/s31/${fn:r:t}.f0 --odir ../../world/s31 ;done
- preprocess.py
- 