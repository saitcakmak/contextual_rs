for i in {0..0}
  do
  wait
  # python main.py config_branin random $i -a
  # python main.py config_branin GP-C-OCBA $i -a
  # python main.py config_branin GP-C-OCBA-1.0 $i -a
  # python main.py config_branin GP-C-OCBA-0.5 $i -a
  python main.py config_branin LEVI-new $i -a
  # python main.py config_hartmann random $i -a
  # python main.py config_hartmann GP-C-OCBA $i -a
  # python main.py config_hartmann GP-C-OCBA-1.0 $i -a
  # python main.py config_hartmann GP-C-OCBA-0.5 $i -a
  python main.py config_hartmann LEVI-new $i -a
  # python main.py config_greiwank random $i -a
  # python main.py config_greiwank GP-C-OCBA $i -a
  # python main.py config_greiwank GP-C-OCBA-1.0 $i -a
  # python main.py config_greiwank GP-C-OCBA-0.5 $i -a
  python main.py config_greiwank LEVI-new $i -a
  python main.py config_cosine8 LEVI-new $i -a
done
