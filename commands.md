
python preprocess.py --text_index 2 --filelists filelists/libritts_audio_sid_text_all_filelist.txt --text_cleaners english_cleaners3

TORIO_USE_FFMPEG_VERSION=5 python train_ms.py -c configs/mb_istft_vits2_libritts.json -m models/mb_istft_vits2_libritts


** From text **

python preprocess.py --text_index 2 --filelists filelists/libritts_audio_sid_grapheme_all_filelist.txt --text_cleaners english_cleaners4

python train_ms.py -c configs/mb_istft_vits2_libritts_text.json -m models/mb_istft_vits2_libritts_from_text