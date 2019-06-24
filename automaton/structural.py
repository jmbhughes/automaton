import numpy as np
from typing import Callable, List, Optional, Tuple
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class StructuralAutomaton:
    def __init__(self, generation0: np.ndarray, initializer: Callable, graph: dict, functions: dict,
                 window_size: int, save_history: bool = True) -> None:
        """
        Initialize a structural automaton. It is defined by the transition graph.
         A transition can be taken if the boolean function on its edge returns true.
        :param generation0: the array of initial states for the automaton.
        :param initializer: when the current state has to grow use this function to initialize the new cells
        :param graph: the transition graph that describes which states can go to which states, e.g.
            {1: [1, 2], 2: [2]} means that state 1 can go to either 1 or 2 while 2 only goes to 2
        :param functions: a nested dictionary of boolean functions that indicate when the transition should be taken as
            a function of the neighbor matrix. Thus a function must have the form f(neighbor): return bool
            For the example given in graph we expect:
            {1: {1: f, 2: g}, 2: {2: h}} where f, g, and h are boolean functions
        :param window_size: will use an odd integer window centered at a coordinate as input for the functions
        :param save_history: whether to save previous iterations in the history
        """
        self.__generation0 = generation0
        self.__initializer = initializer
        self.__graph = graph
        self.__functions = functions
        self.current_iteration = generation0.copy().astype(int)
        self.__next_iteration = self.current_iteration.copy()

        # process the window size
        self.__window_size = window_size
        if self.__window_size <= 0:
            raise RuntimeError("Window size should be a positive odd integer")
        elif self.__window_size % 2 == 0:
            self.__window_size += 1
            raise RuntimeWarning("Window size should be odd, so adding 1.")

        # set up the history as needed
        self.history = save_history
        if self.history:
            self.history = [self.current_iteration.copy()]
        else:
            self.history = None

    def states(self) -> List[int]:
        """
        :return: the list of attainable hugs
        """
        return list(self.__graph.keys())

    def transitions(self) -> List[Tuple[int, int]]:
        """
        :return: the transition edges
        """
        return self.__generate_transitions()

    def __generate_transitions(self) -> List[Tuple[int, int]]:
        """
        :return: the edges of the transition graph.
        Edges are represented as tuples of the state the transition is going from and to.
        """
        edges = []
        for vertex in self.__graph:
            for neighbor in self.__graph[vertex]:
                if (neighbor, vertex) not in edges:
                    edges.append((vertex, neighbor))
        return edges

    def __str__(self) -> str:
        """
        :return: a string representation of the graph
        """
        res = "vertices: "
        for k in self.__graph:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_transitions():
            res += str(edge) + " "
        return res

    def add_state(self, label: int) -> None:
        """
        A new state for the automaton
        :param label: the integer name for the state
        """
        if label not in self.__graph:
            self.__graph[label] = []

    def add_transition(self, edge: Tuple[int, int], function: Callable) -> None:
        """
        Add a possible transition for the automaton
        :param edge: (from, to) being the states the transition is starting from and going into
        :param function: a boolean function to determine if this state should be taken
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.__graph:
            self.__graph[vertex1].append(vertex2)
            self.__functions[vertex1][vertex2] = function
        else:
            self.__graph[vertex1] = [vertex2]
            self.__functions[vertex1] = {vertex2: function}

    def edit_transition(self, edge: Tuple[int, int], function: Callable) -> None:
        """
       Add a possible transition for the automaton
       :param edge: (from, to) being the states the transition is starting from and going into
       :param function: a boolean function to determine if this state should be taken
       """
        if edge[0] in self.__functions:
            if edge[1] in self.__functions[edge[0]]:
                self.__functions[edge[0]][edge[1]] = function
        else:
            self.add_transition(edge, function)

    def __iterate(self) -> None:
        # create a set of patches to iterate over
        patches = view_as_windows(self.current_iteration, (self.__window_size, self.__window_size))
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j]
                state = patch[self.__window_size//2, self.__window_size//2]

                # determine the new state
                for becomes in self.__graph[state]:
                    if self.__functions[state][becomes](patch):
                        self.__next_iteration[i + self.__window_size//2, j + self.__window_size//2] = becomes
                        break

        # switch the current out for this updated version
        self.current_iteration = self.__next_iteration.copy()

        # add to the history if needed
        if self.history:
            self.history.append(self.current_iteration.copy())

    def __expand_grid(self) -> None:
        pass

    def __reached_edge(self) -> None:
        pass

    def iterate(self, n: int = 1) -> None:
        for _ in range(n):
            self.__iterate()

    def show(self, iteration: Optional[int] = None, view_size: Optional[int] = None,
             save: Optional[str] = None, color_table: Optional[dict] = None) -> None:
        if iteration is not None:
            if self.history is False:
                raise RuntimeError("Cannot plot old iteration since history is false.")
            if iteration == 0 or iteration > len(self.history) + 1:
                raise RuntimeError("Requesting a show for an iteration not in history.")
            grid = self.history[iteration - 1]
        else:
            grid = self.current_iteration

        # set up the color map if requested
        if color_table:
            colors = [color_table[i] if i in color_table else "black" for i in range(1 + max(list(color_table.keys())))]
            cmap = ListedColormap(colors)
        else:
            cmap = None

        # make the actual image
        fig, ax = plt.subplots()
        if view_size and view_size < grid.shape[0]:
            offset = (grid.shape[0] - view_size) // 2
            ax.imshow(grid[offset:-offset, offset:-offset], origin='lower', cmap=cmap)
        else:
            ax.imshow(grid, origin="lower", cmap=cmap)

        # refine the image parameters
        ax.set_axis_off()

        # show or save
        if save:
            fig.savefig(save)
            plt.close()
        else:
            fig.show()

    def count(self, label: int) -> int:
        pass


class UlamWarburtonAutomaton(StructuralAutomaton):
    def __init__(self, initial_grid: np.ndarray, save_history: bool = True):
        super().__init__(initial_grid, lambda n: 0,
                         {0: [0, 1], 1: [1]},
                         {0: {0: lambda patch: patch[0, 1] + patch[1, 0] + patch[1, 2] + patch[2, 1] != 1,
                              1: lambda patch: patch[0, 1] + patch[1, 0] + patch[1, 2] + patch[2, 1] == 1},
                          1: {1: lambda patch: True}}, 3, save_history=save_history)


class RegeneratingUWAutomaton(StructuralAutomaton):
    def __init__(self, initial_grid: np.ndarray, delay: int, save_history: bool = True):
        # setup the graph
        graph = {0: [0, 1]}
        for i in range(1, delay):
            graph[i] = [i+1]
        graph[delay] = [0]

        # setup the functions
        functions = {0: {0: lambda patch: patch[0, 1] + patch[1, 0] + patch[1, 2] + patch[2, 1] != 1,
                         1: lambda patch: patch[0, 1] + patch[1, 0] + patch[1, 2] + patch[2, 1] == 1}}
        for i in range(1, delay):
            functions[i] = {i+1: lambda patch: True}
        functions[delay] = {0: lambda patch: True}
        super().__init__(initial_grid, lambda n: 0, graph, functions, 3, save_history=save_history)


class ImmuneUWAutomaton(StructuralAutomaton):
    def __init__(self, initial_grid: np.ndarray, immunity: int, save_history: bool = True):
        # setup the graph
        graph = {0: [0, 1]}
        for i in range(-immunity, 0):
            graph[i] = [i+1]

        # setup the functions
        functions = {0: {0: lambda patch: patch[0, 1] + patch[1, 0] + patch[1, 2] + patch[2, 1] != 1,
                         1: lambda patch: patch[0, 1] + patch[1, 0] + patch[1, 2] + patch[2, 1] == 1}}
        for i in range(-immunity, 0):
            functions[i] = {i + 1: lambda patch: True}
        super().__init__(initial_grid, lambda n: n-immunity if  n < immunity else 0, graph, functions, 3,
                         save_history=save_history)