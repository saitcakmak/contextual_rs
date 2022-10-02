for i in {0..9}
  do
  wait
  # python main.py config_b_3_mean ML_Gao_kde $i -a
  # python main.py config_b_3_mean LEVI $i -a
  # python main.py config_b_3_mean_c ML_Gao $i -a
  # python main.py config_h_4_mean ML_Gao_kde $i -a
  python main.py config_h_4_mean LEVI $i -a
  # python main.py config_h_4_mean_c ML_Gao $i -a
done
