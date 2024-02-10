# Introduce Defects

This repository can be used to invoke 'atom evaporisation' on 2D networks for us with Wilson Group's [2D Network Monte Carlo](https://github.com/WilsonGroupOxford/Network-Monte-Carlo) repository. It is not in its final state, but is essentially complete.

![Flower Network](Flower_Network.png)

## Running the code

To produce a network, run `test_2.py` which creates a network with 36 rings by default, but you can change that number to whatever you like (I've tested networks upwards of 100,000 rings).

It also produces a [LAMMPS](https://www.lammps.org/) data file for the network for use in the [LAMMPS-NetMC](https://github.com/MarshTheBacca/LAMMPS-NetMC) repository.

Then run `introduce_defects.py` to read from these output files, which creates an interactive window where you can click nodes to evaporate them.

Once you are finished evaporating nodes, right click, and the program will form bonds between undercoordinated nodes, and save the network.

This can be done iteratively to introduce several holes in different areas of the network.

## Dependencies

Python 3.12 is required due to the latest type-hinting functionality.
The following python packages need to be installed, for example with `pip install <package_name>`

* numpy
* scipy
* networkx
* matplotlib


## Credit

Credit must be given to [Oliver Whitaker](https://github.com/oliwhitg) for the concept of this repository.
