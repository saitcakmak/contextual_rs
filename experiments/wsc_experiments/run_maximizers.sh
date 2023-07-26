arg_list=(
  "b_3_mean"
  "b_3_worst"
  "g_4_mean"
  "g_4_worst"
  "h_4_mean"
  "c_5_mean"
)

for i in {0..99}
  do
  wait
     python find_maximizers.py config_b_3_mean ML_IKG $i -a
     python find_maximizers.py config_b_3_mean ML_Gao $i -a
done

for i in {0..99}
  do
  wait
     python find_maximizers.py config_g_4_mean ML_IKG $i -a
     python find_maximizers.py config_g_4_mean ML_Gao $i -a
done

for i in {0..99}
  do
  wait
     python find_maximizers.py config_h_4_mean ML_IKG $i -a
     python find_maximizers.py config_h_4_mean ML_Gao $i -a
done

for i in {0..99}
  do
  wait
     python find_maximizers.py config_c_5_mean ML_IKG $i -a
     python find_maximizers.py config_c_5_mean ML_Gao $i -a
done