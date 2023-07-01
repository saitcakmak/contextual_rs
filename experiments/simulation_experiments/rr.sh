for i in {0..1}
  do
  wait
  python main.py config_covid_v4 LEVI-new_f_2 $i -a
done
