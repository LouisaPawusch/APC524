"""

This file contains the basic solver for Conway's Game of Life and will be developed
to add additional levels of complexity to the solver as needed.

The code for this solver was inspired by the cellular automaton forest fires project at 
https://github.com/larantt/clasp410tobiastarsh/tree/main/labs/lab01 and the Game of Life
with 2D Convolution at https://gist.github.com/mikelane/89c580b7764f04cf73b32bf4e94fd3a3

The basic rules of Conway's Game of Life for any type of grid are as follows:
* If a live cell with fewer than two live neighbours exists, it dies of lonliness :(
* If a live cell has two or three live neighbours, it lives :)
* If a live cell has more than three live neighbours, it dies of starvation :(
* If a cell is currently dead and it has exactly three neighbours, a new living cell is born

(add more detail with further development)

TO DO:
- [ ] Further development of the grid function to allow irregular/multidimensional grids
- [ ] Add more complete, cleaner way of specifying rules aside from the default rules function
- [ ] Add a run function to the class with a given number of steps (?)
- [ ] Add a plotting function to the class

"""
