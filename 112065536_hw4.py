import nevergrad as ng
from HomeworkFramework import Function
import numpy as np

class NevergradOptimizer(Function):
    def __init__(self, target_func):
        super().__init__(target_func)
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)
        self.target_func = target_func
        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def objective(self, input_parameters):
        value = self.f.evaluate(self.target_func, input_parameters)
        self.eval_times += 1

        # Check if the returned value is a float
        try:
            value = float(value)
        except ValueError:
            print(f"Error in evaluation: {value}")
            return float('inf')  # Return a large value to indicate error

        if value < self.optimal_value:
            self.optimal_solution[:] = input_parameters
            self.optimal_value = value

        return value

    def run(self, FES):
        optimizer = ng.optimizers.NGO(parametrization=self.dim, budget=FES)
        recommendation = optimizer.minimize(self.objective)
        return self.get_optimal()

if __name__ == '__main__':
    func_num = 1
    fes = 0
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500

        op = NevergradOptimizer(func_num)
        best_input, best_value = op.run(fes)
        
        print(best_input, best_value)
        
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1
