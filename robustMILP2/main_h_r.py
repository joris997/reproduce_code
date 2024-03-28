import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('/home/joris/Documents/code/MAMP_ET_Int')

from planning.robustMILP2.bezier_STL_MILP import BezierSTL_MILP
from planning.robustMILP2.STL_specs import get_stl_spec


if __name__ == "__main__":
    if True:
        # get the STL specification, this also contains
        # the t0, tf, N, and order of the Bezier curves
        spec, robot, world = get_stl_spec(num=7)

        BSTL = BezierSTL_MILP(spec,robot,world)
        BSTL.construct()
        BSTL.add_cost()
        BSTL.solve()
        BSTL.plot(path="planning/robustMILP2/figures/bstl.png")
        # BSTL.plot()


    if False:
        # running comparison tests
        # Define the range of your parameters

        ### for UAV altitude control
        N_values = range(3,11)
        order_values = range(4,11)
        ### for surveillance
        # N_values = range(6, 11) 
        # order_values = range(3, 7)

        # Initialize arrays to hold the results
        solve_times = np.zeros((len(N_values), len(order_values)))
        num_vars = np.zeros((len(N_values), len(order_values)))
        num_bin_vars = np.zeros((len(N_values), len(order_values)))
        num_int_vars = np.zeros((len(N_values), len(order_values)))
        num_constrs = np.zeros((len(N_values), len(order_values)))

        theta_ps = np.zeros((len(N_values), len(order_values)))
        theta_ms = np.zeros((len(N_values), len(order_values)))

        # Loop over all combinations of N and order
        for i, N in enumerate(N_values):
            for j, order in enumerate(order_values):
                
                Ntests = 5
                for _ in range(Ntests):
                    spec, robot, world = get_stl_spec(num=4,N=N,order=order)

                    BSTL = BezierSTL_MILP(spec,robot,world)
                    BSTL.construct()
                    BSTL.add_cost()
                    BSTL.solve()
                    BSTL.evaluate()

                    # Record the results
                    solve_times[i, j] += BSTL.solve_time
                    num_vars[i, j] = BSTL.num_vars
                    num_bin_vars[i, j] = BSTL.num_bin_vars
                    num_int_vars[i, j] = BSTL.num_int_vars
                    num_constrs[i, j] = BSTL.num_constrs

                    theta_ms[i, j] = BSTL.spec.theta_m_sol
                    theta_ps[i, j] = BSTL.spec.theta_p_sol

                solve_times[i,j] /= Ntests
                print("Solve time: ", solve_times[i,j])
        # Save results into DataFrame
        df = pd.DataFrame({
            'N': np.repeat(N_values, len(order_values)),
            'Order': np.tile(order_values, len(N_values)),
            'Solve Time': solve_times.flatten(),
            'Num Variables': num_vars.flatten(),
            'Num Binary Variables': num_bin_vars.flatten(),
            'Num Integer Variables': num_int_vars.flatten(),
            'Num Constraints': num_constrs.flatten(),
            'Theta_m': theta_ms.flatten(),
            'Theta_p': theta_ps.flatten()
        })

        # Save the DataFrame to a CSV file
        df.to_csv('planning/robustMILP2/figures/simulation_results.csv', index=False)

        # Create the plots
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        # Plot solve times
        cax1 = ax[0,0].imshow(solve_times.T, interpolation='nearest', origin='lower', cmap='summer')
        ax[0,0].set_title('Solve Times')
        ax[0,0].set_xticks(range(len(N_values)))
        ax[0,0].set_xticklabels(N_values)
        ax[0,0].set_xlabel('N')
        ax[0,0].set_yticks(range(len(order_values)))
        ax[0,0].set_yticklabels(order_values)
        ax[0,0].set_ylabel('Order')
        fig.colorbar(cax1, ax=ax[0,0])

        # Plot number of variables
        cax2 = ax[0,1].imshow(num_vars.T, interpolation='nearest', origin='lower', cmap='summer')
        ax[0,1].set_title('Number of Variables')
        ax[0,1].set_xticks(range(len(N_values)))
        ax[0,1].set_xticklabels(N_values)
        ax[0,1].set_xlabel('N')
        ax[0,1].set_yticks(range(len(order_values)))
        ax[0,1].set_yticklabels(order_values)
        ax[0,1].set_ylabel('Order')
        fig.colorbar(cax2, ax=ax[0,1])

        cax3 = ax[1,0].imshow(num_constrs.T, interpolation='nearest', origin='lower', cmap='summer')
        ax[1,0].set_title('Number of Constraints')
        ax[1,0].set_xticks(range(len(N_values)))
        ax[1,0].set_xticklabels(N_values)
        ax[1,0].set_xlabel('N')
        ax[1,0].set_yticks(range(len(order_values)))
        ax[1,0].set_yticklabels(order_values)
        ax[1,0].set_ylabel('Order')
        fig.colorbar(cax3, ax=ax[1,0])

        cax4 = ax[1,1].imshow(theta_ps.T, interpolation='nearest', origin='lower', cmap='summer')
        ax[1,1].set_title('Positive Temp Robustness')
        ax[1,1].set_xticks(range(len(N_values)))
        ax[1,1].set_xticklabels(N_values)
        ax[1,1].set_xlabel('N')
        ax[1,1].set_yticks(range(len(order_values)))
        ax[1,1].set_yticklabels(order_values)
        ax[1,1].set_ylabel('Order')
        fig.colorbar(cax4, ax=ax[1,1])

        # Show the plots
        plt.show()

