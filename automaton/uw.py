from __future__ import annotations
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from typing import Optional


class UlamWarburtonWait:
    def __init__(self, history: bool = True, initial_grid: Optional[np.ndarray] = None,
                 initial_size: int = 501,
                 death_time: int = np.inf, wait_time: int = 1, initially_dead: bool = False) -> None:
        """
        Creates an automaton that includes the death timer.

        Parameters
        ----------
          history      : bool
                         whether to save historical iterations of the automaton

          initial_grid : np.array
                         a square array to use as the initial grid

          initial_size : int
                         An odd integer that is used to initialize the grid when no
                         initial_grid is given. The center square is activated.

          death_time   : int
                         After being active for this many iterations,
                         the cell deactivates.
          wait_time    : int
                         How long a cell must remain inactive before reactivating
                         after death
        initially_dead : bool
                         If True, inactive cells are initially dead and have to wait.
                         If False, inactive cells start out without a wait time.
        """
        self.iteration = 0

        # set up the grid
        if initial_grid is not None:  # use a passed grid
            if len(initial_grid.shape) != 2:
                msg = "intitial_grid must be 2 dimensional "
                raise ValueError(msg)
            if initial_grid.shape[0] != initial_grid.shape[1]:
                msg = "intial_grid must be square but is {}".format(initial_grid.shape)
                raise ValueError(msg)
            self.grid = initial_grid.copy()
        else:  # no passed grid so use size
            if initial_size % 2 == 0:
                msg = "Initial size should be odd, but it is {}".format(initial_size)
                raise ValueError(msg)
            self.grid = np.zeros((initial_size, initial_size), dtype=bool)
            self.grid[initial_size // 2, initial_size // 2] = 1

        # set up the history logging if needed
        if history:
            self.history = [self.grid.copy()]
        else:
            self.history = []

        # since the window remains constant, define it once. This is the cross
        # window so it only examines north, west, east, and south
        self.windows = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        # the portion relating to death
        self.age = np.zeros(self.grid.shape, dtype=int)
        self.death_time = death_time

        # the portion relating to waiting
        self.initially_dead = initially_dead
        self.wait_time = wait_time
        # start at one since the iterate command immediately decrements
        self.wait_age = np.zeros(self.grid.shape, dtype=int) + 1
        if self.initially_dead:  # make all the dead cells wait if requested
            self.wait_age[np.logical_not(self.grid)] = self.wait_time

    def show(self, iteration: Optional[int] = None,
             save: Optional[str] = None, view_size: Optional[int] = None) -> None:
        """
        Display the grid.

        Parameters
        ----------
        iteration : integer or None
                    The index (starting at 1) to show.
                    Integers are only possible if history is being logged.
                    If None, it just shows the most recent grid.
        save      : str or None
                    Where to save the image. If None, it does not save.
        view_size : integer or None
                    If None, shows the full grid.
                    Otherwise shows the view_size centermost pixel square.

        Examples
        --------
        uw = UlamWarburton()
        uw.iterate(steps=100)
        uw.show()
        uw.show(iteration=40)
        uw.show(save="/home/media/drive/img.png")
        """

        # get the specific iteration to show
        if iteration is not None:
            if iteration == 0 or iteration > len(self.history) + 1:
                raise RuntimeError("Requesting a show for an iteration not in history.")
            grid = self.history[iteration - 1]
        else:
            grid = self.grid

        # make the actual image
        fig, ax = plt.subplots()
        if view_size and view_size < grid.shape[0]:
            offset = (grid.shape[0] - view_size) // 2
            ax.imshow(grid[offset:-offset, offset:-offset], origin='lower')
        else:
            ax.imshow(grid, origin="lower")

        # refine the image parameters
        ax.set_axis_off()

        # show or save
        if save:
            fig.savefig(save)
            plt.close()
        else:
            fig.show()

    def _iterate(self) -> None:
        """
        An internal method to handle a single iteration.
        """

        # make sure the grid hasn't been outgrown
        if self._reached_edge():
            self._expand_grid()

        self.age[self.grid] += 1
        self.wait_age[np.logical_not(self.grid)] -= 1

        # do the update
        grow_mask = (scipy.signal.convolve2d(self.grid,
                                             self.windows,
                                             mode='same') == 1)
        grow_mask *= (self.wait_age <= 0)  # don't let waiting cells grow

        self.grid[grow_mask] = True
        self.age[grow_mask] = 1

        death_mask = (self.age > self.death_time)
        self.age[death_mask] = 0
        self.grid[death_mask] = False
        self.wait_age[death_mask] = self.wait_time

        self.iteration += 1
        if self.history:
            self.history.append(self.grid.copy())

    def iterate(self, steps: int = 1) -> None:
        """
        Public method to iterate the automaton.

        Parameters
        ----------
        steps     : int
                    How many steps to advance by

        Examples
        --------
        uw = UlamWarburton()
        uw.iterate(steps=100)
        uw.iterate()
        """
        for _ in range(steps):
            self._iterate()

    def _expand_grid(self) -> None:
        """
        Increase the size of the grid because the edge has been reached.
        """
        # increase the current grid always
        old_size = self.grid.shape[0]
        new_size = (2 * old_size) + 1
        extra_index = (new_size - old_size) // 2

        new_grid = np.zeros((new_size, new_size), dtype=bool)
        new_grid[extra_index:-extra_index, extra_index:-extra_index] = self.grid
        self.grid = new_grid.copy()

        new_age = np.zeros((new_size, new_size), dtype=int)
        new_age[extra_index:-extra_index, extra_index:-extra_index] = self.age
        self.age = new_age.copy()

        new_wait_age = np.zeros((new_size, new_size), dtype=int)
        universe_waiting = (self.wait_time - self.iteration)
        universe_waiting = 0 if universe_waiting < 0 else universe_waiting
        new_wait_age += universe_waiting
        new_wait_age[extra_index:-extra_index, extra_index:-extra_index] = self.wait_age
        self.wait_age = new_wait_age.copy()

        # increase the history grid if needed
        if self.history:
            new_history = []
            for old in self.history:
                new_grid = np.zeros((new_size, new_size), dtype=bool)
                new_grid[extra_index:-extra_index, extra_index:-extra_index] = old
                new_history.append(new_grid.copy())
            self.history = new_history

    def _reached_edge(self) -> bool:
        """
        Determine if the grid edge has been reached

        Returns
        -------
        bool   : True if edge reached.
        """
        bounds = [self.grid[0, :],
                  self.grid[:, 0],
                  self.grid[-1, :],
                  self.grid[:, -1]]
        bounds = np.concatenate(bounds)
        return np.any(bounds)
