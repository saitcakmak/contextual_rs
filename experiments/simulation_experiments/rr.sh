for i in {0..4}
  do
  wait
  python main_ts.py config_covid TS $i -a
  wait
  python main_ts.py config_covid TS+ $i -a
  wait
  python main.py config_covid DSCO $i -a
  wait
  python main.py config_covid C-OCBA $i -a
  wait
  python main.py config_covid GP-C-OCBA $i -a
done
