for i in {0..29}
  do
#  wait
#  python main_ts.py config_covid TS $i -a
#  wait
#  python main_ts.py config_covid TS+ $i -a
  wait
  python main.py config_covid_v2 GP-C-OCBA_f_2 $i -a
#  wait
#  python main.py config_covid C-OCBA $i -a
#  wait
#  python main.py config_covid GP-C-OCBA $i -a
done
