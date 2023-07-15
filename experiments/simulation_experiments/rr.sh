for i in {0..49}
  do
  wait
#  python main.py config_covid_v4 LEVI-new_f_2 $i -a
  python main.py config_covid_v4 GP-C-OCBA-new_f_2 $i -a
done
