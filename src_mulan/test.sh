# python main.py test --input ../data/test.txt_1w_char_seg --model ../modeltest

#python main.py test --input ../data/test.txt.banjiao_1w_char_seg --model ../config_50dim_1layer

#python main.py test --input ../data/test.txt.banjiao_1w_char_seg --model ../config_50dim_2layer

#python main.py test --input ../data/test.txt.banjiao_1w_char_seg --model ../config_50dim_3layer

#python main.py test --input ../data/test.txt.banjiao_1w_char_seg --model ../config_50dim_4layer

#python main.py test --input ../data/test.txt.banjiao_1w_char_seg --model ../config_300dim_1layer

#python main.py test --input ../data/test.txt.banjiao_1w_char_seg --model ../config_300dim_2layer

#python main.py test --input ../data/test.txt.banjiao_1w_char_seg --model ../config_300dim_3layer

#python main.py test --input ../data/test.txt.banjiao_1w_char_seg --model ../config_300dim_4layer 

# 2018-10-14 19:39:51,264 INFO: Test Result : ppl    83.53 | ppl_back   84.20 
# 2018-10-14 19:40:26,789 INFO: Test Result : ppl    80.15 | ppl_back   80.45 
# 2018-10-14 19:41:15,663 INFO: Test Result : ppl    81.24 | ppl_back   80.38 
# 2018-10-14 19:42:16,008 INFO: Test Result : ppl    85.18 | ppl_back   85.58 
# 2018-10-14 19:42:38,087 INFO: Test Result : ppl    37.06 | ppl_back   37.00 
# 2018-10-14 19:43:14,512 INFO: Test Result : ppl    36.35 | ppl_back   35.94 
# 2018-10-14 19:44:03,974 INFO: Test Result : ppl    37.31 | ppl_back   37.23 
# 2018-10-14 19:45:08,770 INFO: Test Result : ppl    37.74 | ppl_back   37.75  

#python main.py train  --lr 0.0001 --lr_decay 0.8 --bptt 40 --gaz_file ../data/emb/lm_mincount3_gaz200_len234_with_zeros --max_epoch 15 --train_path ../data/train1m --dev_path ../data/dev10k --config_path config_300dim_1layer.json --model ../mulan_config_300dim_1layer

#python main.py train  --lr 0.0001 --lr_decay 0.8 --bptt 40 --gaz_file ../data/emb/lm_mincount3_gaz200_len234_with_zeros --max_epoch 15 --train_path ../data/train1m --dev_path ../data/dev10k --config_path config_300dim_2layer.json --model ../mulan_config_300dim_2layer

python main.py test --test_path ../data/test5k --model ../mulan_config_300dim_2layer

