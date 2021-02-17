## This directory contains the experiments (obviously).

### It is structured as follows:
- Each experiment type has its own sub-directory.
- Within this sub-directory, the main Python script for this experiment is found. 
- The script is runnable with `python main.py <dirname> <seed>`.
- `<dirname>` specifies the location for both the configuration and output files.
- The experiment script will read its configuration from `<dirname>/config.json`.
- `config.json` includes a single dictionary (call it `kwargs`) with simple Python objects.
- The main function of script should be called as `main(seed=seed, **kwargs)`.
- The script will write its output file in `<dirname>/<seed>.pt` where `<seed>` is 
  written with 4 digits with as many zeros filling in as needed.

- See below for the script for reading the config, running `main(...)` and saving the 
  output. It also checks whether the output exists. Pass `-f` as a 3rd argument to 
  overwrite existing output.
```python
if __name__ == '__main__':
    current_dir = path.dirname(path.abspath(__file__))
    exp_dir = path.join(current_dir, sys.argv[1])
    config_path = path.join(exp_dir, "config.json")
    seed = int(sys.argv[2])
    output_path = path.join(exp_dir, f"{str(seed).zfill(4)}.pt")
    if path.exists(output_path):
        if len(sys.argv) > 3 and sys.argv[3] == "-f":
            print("Overwriting the existing output!")
        else:
            print(
                "The output file exists for this experiment & seed!"
                "Pass -f as the 3rd argument to overwrite!"
            )
            quit()
    with open(config_path, "r") as f:
        kwargs = json.load(f)
    output = main(seed=seed, **kwargs)
    torch.save(output, output_path)
```

- Bash script for running seeds 0 to 9 in a for loop:
```bash
for i in {0..9}; do python main.py <dirname> $i; done
```

- Bash script for running seeds 20 to 29 with batches of 4 in parallel:
```bash
N=4; for i in {20..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py <dirname> $i & done
```
