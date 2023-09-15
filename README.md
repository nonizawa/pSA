# pSA (p-bit based simulated annealing)

[![GitHub license](https://img.shields.io/github/license/nonizawa/pSA)](https://github.com/nonizawa/pSA/blob/main/LICENSE)

PSA is an implementation of simulated annealing using p-bits. pSA has an issue for solving large-scale combinatorial optimization problems due to an unexpected oscillation among p-bits. To address this challenge, we propose two novel algorithms, time average pSA (TApSA) and stalled pSA (SpSA). These algorithms are designed based on partial deactivation of p-bits and are thoroughly tested using Python simulations on maximum cut benchmarks that are typical combinatorial optimization problems. On the 16 benchmarks from 800 to 5,000 nodes, the proposed methods improve the normalized cut value from 0.8% to 98.4% on average in comparison with the conventional pSA.

## Installation

### Prerequisites

- Python 3.x

### Clone the Repository

To get started, clone the repository using git:

```sh
git clone https://github.com/nonizawa/pSA.git
cd pSA
```

## Structure

- `TApSA.py`: This is the Python script that runs the TApSA for MAX-CUT algorithm.
- `SpSA.py`: This is the Python script that runs the SpSA for MAX-CUT algorithm.
- `./graph/`: This directory contains the dataset of graphs used for evaluation.
- `./result/`: This directory contains the evaluation results generated using simulation.

## Single Run

### TApSA (Time Averaged pSA)

To run the SSA algorithm on a single instance, use the sa.py script. For example:

```sh
python ssa.py --file_path graph/G1.txt --cylce 1000 --trial 100 --tau 1 --param 1
```
You can find the simulation results in ./result/.　Result***.csv includes simulation retsuls, such as mean cut values and simulation time. Cut***.csv includes cut values for all trials.

Here ia the options.

- `--graph_file`: a graph file

- `--cycle`: Number of cycles for 1 trial

- `--trial`: Number of trials to evaluate the performance on average

- `--tau`:  A pseudo inverse temperature is increased every tau cycle

- `--mean_range`: The size of time window for averaing 

### SpSA (Stalled pSA)

Here is a python program for SSAU that can run, like SSA.
- `SpSA.py`

It has the same options as TApSA except the folling one:

- `--stall_prop`: The probability of p-bits stalled

## Contact

For any questions, issues, or inquiries, feel free to create an issue in the repository or contact the repository owner [@nonizawa](https://github.com/nonizawa).

## Citation

## License

This project is licensed under the MIT License.
