# WGPU FDM
A simple library crate that uses WGPU to run a finite differences simulation of
a string on
a GPU. This simulation oversamples 2:1 – this is done so stability can be
achieved for higher wave speeds. Functionality may be extended in the near
future :))

## Example
This crate comes with an example going over the basic functionality.
It can be run using
```bash
cargo run --example example
```
and will output samples to console.

To view and plot the displacement as a function of time one could pipe the
output into a file and use a simple python script, for example:
```python
import matplotlib.pyplot as plt

def read_and_concatenate_lists(file_path):
    concatenated_list = []
    with open(file_path, 'r') as file:
        for line in file:
            line_list = eval(line.strip())
            concatenated_list.extend(line_list)
    return concatenated_list

def plot_list(data):
    plt.plot(data)
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Concatenated List over Time')
    plt.show()

file_path = 'example_data'
data = read_and_concatenate_lists(file_path)
plot_list(data)
```
```bash
cargo run --examples example > example_data && python <script-name-here>
```
