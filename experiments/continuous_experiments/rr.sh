for i in {0..29}
  do
  wait
  python main.py config_branin LEVI-new $i -a
  python main.py config_hartmann LEVI-new $i -a
  python main.py config_greiwank LEVI-new $i -a
  python main.py config_cosine8 LEVI-new $i -a
done

for i in {30..49}
  do
  wait
  python main.py config_branin LEVI-new $i -a
  python main.py config_hartmann LEVI-new $i -a
  python main.py config_greiwank LEVI-new $i -a
  python main.py config_cosine8 LEVI-new $i -a
done
#
#for i in {50..99}
#  do
#  wait
#  python main.py config_branin LEVI-new $i -a
#  python main.py config_hartmann LEVI-new $i -a
#  python main.py config_greiwank LEVI-new $i -a
#  python main.py config_cosine8 LEVI-new $i -a
#done