import numpy as np
from typing import Callable, List, Optional, Tuple
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph


class StructuralAutomaton:
    def __init__(self, generation0: np.ndarray, initializer: Callable, graph: dict, functions: dict,
                 window_size: int, save_history: bool = True) -> None:
        """
        Initialize a structural automaton. It is defined by the transition graph. This indicates which states can
        go to which states. A transition in the graph can be taken if the boolean function on its edge returns true.
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
        self.generation = 0

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
        """
        A helper function to complete one iteration
        """
        self.generation += 1
        # create a set of patches to iterate over
        patches = view_as_windows(self.current_iteration, (self.__window_size, self.__window_size))

        # for each patch
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                # look at that patch
                patch = patches[i, j]
                state = patch[self.__window_size//2, self.__window_size//2]  # the center of the patch

                # determine the new state using the transition functions
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
        """
        Iterate the automaton forward
        :param n: how many steps forward to go
        """
        for _ in range(n):
            self.__iterate()

    def show(self, iteration: Optional[int] = None, view_size: Optional[int] = None,
             save: Optional[str] = None, color_table: Optional[dict] = None, show_title: bool = False) -> None:
        """
        Show the automaton using Matplotlib as an image
        :param iteration: which iteration to show
        :param view_size: If None, shows the full grid.
            Otherwise shows the view_size pixel square centered on the middle..
        :param save:  Where to save the image. If None, it does not save.
        :param color_table: How to color the automaton. Give a dictionary with keys of states and values of colors.
            If None, uses Matplotlib's default
        :param show_title: whether to print the iteration number as a title
        """
        # find the iteration
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
            colors = [color_table[i] if i in color_table else "black" for i in range(min(list(color_table.keys())),
                                                                                     1 + max(list(color_table.keys())))]
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
        if show_title:
            ax.set_title("i={}".format(iteration))

        fig.tight_layout()
        # show or save
        if save:
            fig.savefig(save)
            plt.close()
        else:
            fig.show()

    def count(self, labels: List[int], iteration: Optional[int] = None) -> int:
        """
        Count the number of cells in an iteration that have the specified labels
        :param labels: which labels to count
        :param iteration: the iteration to look at. If None, uses the current iteration.
        :return: Number of cells with that label.
        """
        grid = self.current_iteration if iteration is None else self.history[iteration]
        return sum([np.sum(grid == label) for label in labels])

    def draw_graph(self, path: str) -> None:
        """
        Draws the transition graph using networkx and graphviz
        :param path: where to save the image
        """
        g = nx.MultiDiGraph()
        g.add_nodes_from(self.states())
        g.add_edges_from(self.transitions())
        a = to_agraph(g)
        a.layout("dot")
        a.draw(path)


class UlamWarburtonAutomaton(StructuralAutomaton):
    """
    The standard Ulam Warburton automaton. This automaton works by using a window of 3x3. If exactly one cell
    to the north, south, west, or east is on, then the cell turns on. Once a cell is on, it never turns off.
    """
    def __init__(self, initial_grid: np.ndarray, save_history: bool = True):
        """
        Create an Ulam Warburton Automaton
        :param initial_grid: The grid to begin with
        :param save_history: whether to save iterations
        """
        super().__init__(initial_grid, lambda n: 0,
                         {0: [0, 1], 1: [1]},
                         {0: {0: lambda patch: patch[0, 1] + patch[1, 0] + patch[1, 2] + patch[2, 1] != 1,
                              1: lambda patch: patch[0, 1] + patch[1, 0] + patch[1, 2] + patch[2, 1] == 1},
                          1: {1: lambda patch: True}}, 3, save_history=save_history)


class RegeneratingUWAutomaton(StructuralAutomaton):
    """
    Similar to the Ulam Warburton automaton except now on cells turn back off after being on for a specified number
    of iterations. We do this by introducing new states that transition from the on.
    """
    def __init__(self, initial_grid: np.ndarray, delay: int, save_history: bool = True):
        """
        Create a Regenerating Automaton
        :param initial_grid: the first generation state
        :param delay: how many generations an on cell stays on
        :param save_history: whether to save iterations
        """
        # setup the graph
        graph = {0: [0, 1]}
        for i in range(1, delay):
            graph[i] = [i+1]
        graph[delay] = [0]

        # setup the functions
        functions = {0: {0: lambda patch: np.sum(np.sign(np.array([patch[0, 1], patch[1, 0],
                                                                   patch[1, 2], patch[2, 1]])) > 0) != 1,
                         1: lambda patch: np.sum(np.sign(np.array([patch[0, 1], patch[1, 0],
                                                                   patch[1, 2], patch[2, 1]])) > 0) == 1}}
        for i in range(1, delay):
            functions[i] = {i+1: lambda patch: True}
        functions[delay] = {0: lambda patch: True}
        super().__init__(initial_grid, lambda n: 0, graph, functions, 3, save_history=save_history)


class ImmuneUWAutomaton(StructuralAutomaton):
    """
    A very silly version of the Ulam Warburton automaton where cells never wait a delay time before turning on.
    We do the same thing as in RegeneratingAutomaton
    """
    def __init__(self, initial_grid: np.ndarray, delay: int, save_history: bool = True):
        """
        Create an Immune Ulam Warburton Automaton
        :param initial_grid: the first generation state
        :param delay: how many generations an off cell must be off before activating
        :param save_history: whether to save iterations
        """
        # setup the graph
        graph = {0: [0, 1]}
        for i in range(-delay, 0):
            graph[i] = [i+1]

        # setup the functions
        functions = {0: {0: lambda patch: np.sum(np.sign(np.array([patch[0, 1], patch[1, 0],
                                                                   patch[1, 2], patch[2, 1]])) > 0) != 1,
                         1: lambda patch: np.sum(np.sign(np.array([patch[0, 1], patch[1, 0] +
                                                                   patch[1, 2], patch[2, 1]])) > 0) == 1}}
        for i in range(-delay, 0):
            functions[i] = {i + 1: lambda patch: True}
        super().__init__(initial_grid, lambda n: n-delay if n < delay else 0, graph, functions, 3,
                         save_history=save_history)


class ImmuneRegeneratingUWAutomaton(StructuralAutomaton):
    """
    A combination between the ImmuneUWAutomaton and RegeneratingUWAutomaton. Cells both have a waiting period to
    turn on and to turn off. We do this by lots of extra nodes. :D
    """
    def __init__(self, initial_grid: np.ndarray, on_period: int, off_period: int, save_history: bool = True):
        """
        Create this kind of automaton
        :param initial_grid: the initial grid
        :param on_period: how long an on cell stays on before deactivating
        :param off_period: how long an off cell stays off before deactivating
        :param save_history: whether to save previous iterations
        """
        # setup the graph
        # basic UW
        graph = {0: [0, 1]}

        # these are the additional states for on_period generations before shutting back off
        for i in range(1, on_period):
            graph[i] = [i+1]
        graph[on_period] = [-off_period]

        # these are the additional states for off_period generations where a cell is immune and won't turn on
        for i in range(-off_period, 0):
            graph[i] = [i+1]

        # setup th functions
        # the basic UW
        functions = {0: {0: lambda patch: np.sum(np.sign(np.array([patch[0, 1], patch[1, 0],
                                                                   patch[1, 2], patch[2, 1]])) > 0) != 1,
                         1: lambda patch: np.sum(np.sign(np.array([patch[0, 1], patch[1, 0],
                                                                   patch[1, 2], patch[2, 1]])) > 0) == 1}}

        # functions for on transitions
        for i in range(1, on_period):
            functions[i] = {i+1: lambda patch: True}
        functions[on_period] = {-off_period: lambda patch: True}

        # functions for off transitions
        for i in range(-off_period, 0):
            functions[i] = {i+1: lambda patch: True}
        super().__init__(initial_grid, lambda n: n-off_period if n < off_period else 0, graph, functions, 3,
                         save_history=save_history)
