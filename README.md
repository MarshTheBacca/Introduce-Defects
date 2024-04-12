# Introduce Defects

This repository can be used to invoke 'atom evaporisation' on 2D networks for use with Wilson Group's [Bond Switch Simulator](https://github.com/MarshTheBacca/Bond-Switch-Simulator) repository.

![Flower Network](Flower_Network.png)

## Running the code

Running `main.py` will present you with 5 options. You can either:

### Introduce Defects

You can introduce defects into either a new network or an existing network (which can be done iteratively, to introduce several defects).
An interactive window will appear, where you can click nodes to delete them. When you are finished deleting nodes, right click, and the program will attempt to form bonds between undercoordinated nodes. If you have not met the following criteria, you will have to continue deleting nodes:

* There are an odd number of undercoordinated nodes
* There are 3 or more undercoordinated nodes adjacent to one another
* There are undercoordinated nodes that are members of different rings
* There are an odd number of nodes separating 'islands'

These contraints are necessary to fill coordination of all nodes properly. An _island_ is when two undercoordinated nodes are adjacent to one another.

Upon adhering to these constraints, the program will save the network, as well as a [LAMMPS](https://www.lammps.org/) data file for the network, which is needed for the Bond Switch Simulator.

### Visualise a Network

Self explanatory, you choose from a list of saved networks, and the program will display the network using the new 'pretty plotting' function. Fixed rings are drawn in red, rings over a size of 10 are drawn in white

### Delete a Network

Self explanators, you choose from a list of saved networks, and the program will delete it

### Copy a Network

Self explanators, you choose from a list of saved networks, and the program will copy it with a given new name

### Create a fixed_rings.txt File

The fixed_rings.txt file is used by _Bond Switch Simulator_ to make a ring invariable during a simulation. This is supposed to represent a defect that has been introduced via templating.

## Dependencies

Python 3.12 is required due to the latest type-hinting functionality.
The following python packages need to be installed, for example with `pip install <package_name>`

* numpy
* scipy
* networkx
* matplotlib


## Credit

Credit must be given to [Oliver Whitaker](https://github.com/oliwhitg) for the concept of this repository.
