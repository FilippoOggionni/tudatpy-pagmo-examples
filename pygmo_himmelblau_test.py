import math
from pygmo import *

class HimmelblauOptimization:

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def get_bounds(self):
        """ Define the search space """
        return ([self.x_min,self.y_min], [self.x_max,self.y_max])

    def fitness(self, x: list):
        function_value = math.pow(x[0] * x[0] + x[1] - 11.0, 2.0) + math.pow(x[0] + x[1] * x[1] - 7.0, 2.0)
        return [function_value]

def main():

    pop_size = 1000

    algo = algorithm( de( gen=1 ) )
    prob = problem(HimmelblauOptimization( -5.0, 5.0, -5.0, 5.0 ) )
    pop = population(prob, size=pop_size)

    for i in range(100):
        pop = algo.evolve(pop)
    print(pop.champion_f)
    print(pop.champion_x)

    return 0


if __name__ == "__main__":
    main()
