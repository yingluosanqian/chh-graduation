import matplotlib.pyplot as plt
import numpy as np


def load_result(filepath):
    ls = []
    with open(filepath, 'r') as f:
        for line in f:
            if 'geo_mean' in line and 'total' not in line:
                line = line.split(' ')[0]
                value = float(line.split('=')[1])
                ls.append(value)
    return ls


def work(data_bs, data_rs1, data_rs2):
    data_bs = np.array(data_bs)
    data_rs1 = np.array(data_rs1)
    data_rs2 = np.array(data_rs2)

    # Set up the x-axis values (iterations 1 to 50)
    iterations = np.arange(1, 51)

    # Create the plot
    plt.figure(figsize=(11, 7))
    plt.plot(iterations, data_bs, label='bpc coreset(Ours)', marker='o')
    plt.plot(iterations, data_rs1, label='rs coreset 1', marker='s')
    plt.plot(iterations, data_rs2, label='rs coreset 2', marker='^')


    # Adding labels and title
    # plt.title('Comparison of Optimization Methods Over Iterations')
    plt.xlabel('Top k', fontsize=20)
    plt.ylabel('GMeanOverOz', fontsize=20)

    plt.legend(loc='lower right', fontsize='20', frameon=True, shadow=True, borderpad=1)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    # Show grid
    plt.grid(True)

    # Display the plot
    # plt.show()
    plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    data_bs = load_result(f'evaluation_result_final_mlp_ylsq_mlp.txt')
    data_rs1 = load_result(f'evaluation_result_final_mlp_rs_coreset_1_mlp.txt')
    data_rs2 = load_result(f'evaluation_result_final_mlp_coreset_mlp.txt')
    work(data_bs, data_rs1, data_rs2)
    pass
