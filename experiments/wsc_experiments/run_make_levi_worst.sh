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
     python make_levi_new_worst.py config_b_3_worst config_b_3_mean LEVI_new $i
     python make_levi_new_worst.py config_g_4_worst config_g_4_mean LEVI_new $i
done