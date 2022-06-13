from utils.tests import *
from utils.Logger import Logger
from utils.utils import gradient, benchmark
import matplotlib.pyplot as plt


Logger.ASSERTION_EXIT = False
Logger.ENABLE = False


# //////// INPUT DATA

fn = lambda x1, x2: (10 * (x1 - x2) ** 2 + (x1 - 1) ** 2) ** 4
start_point = np.array([-1.2, 0.0])


def task1():

    bounds = [1, 16]

    @benchmark
    def eps_test(eps=3, h=5, criteria=3):
        grad = lambda point: gradient(fn, point, 10**-h)

        params = {
            "one_dim_method": "golden_section",
            "grad": grad,
            "step": None,
            "one_dim_eps": 10**-eps,
            "criteria_eps": 10**-criteria
        }

        result = GradientDescent(fn, start_point, **params).start()
        return result

    def set_graphic(ax: plt.axes, values, title):
        print(title, values)    # , values[:2], list(map(lambda x: f"{x:.10f}", values[2])))
        ax.plot(values[0], values[1], marker='o')
        ax.set_ylabel("кількість ітерацій")
        ax.set_xlabel("точність")
        ax.set_title(title)
        ax.set_xticks(values[0], [f"1e-{i}" for i in values[0]])
        for i, (x, y, _) in enumerate(zip(*values)):
            ax.annotate(y, (x, y))
        ax.set_ylim(0, 60)
        ax.grid(True)

    fig, axes = plt.subplots(4, 1, figsize=(9, 12))

    print("\ntest gradient\n")
    grad_eps_test = [[],[], []]
    for i in range(*bounds):
        result = eps_test(h=i, eps=3, criteria=3)
        grad_eps_test[0].append(i)
        grad_eps_test[1].append(result.iterations)
        grad_eps_test[2].append(result.f(*result.x))

    print("\neps test\n")
    gd_eps_test = [[],[], []]
    for i in range(*bounds):
        result = eps_test(eps=i)
        gd_eps_test[0].append(i)
        gd_eps_test[1].append(result.iterations)
        gd_eps_test[2].append(result.f(*result.x))

    print("\ncriteria eps test\n")
    criteria_eps_test = [[],[], []]
    for i in range(*bounds):
        result = eps_test(criteria=i)
        criteria_eps_test[0].append(i)
        criteria_eps_test[1].append(result.iterations)
        criteria_eps_test[2].append(result.f(*result.x))

    print("\ngradient and epses test\n")
    gd_grad_eps_test = [[],[], []]
    for i in range(*bounds):
        result = eps_test(eps=i, h=i, criteria=i)
        gd_grad_eps_test[0].append(i)
        gd_grad_eps_test[1].append(result.iterations)
        gd_grad_eps_test[2].append(result.f(*result.x))

    # grad_eps_test = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [200, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 4], [1.1138654959679026, 1.9338322273837666e-12, 1.9016646035506197e-06, 2.3389657061528475e-07, 4.451843947909425e-07, 2.610362292855971e-07, 2.467447660342118e-07, 2.4539476994885313e-07, 2.4498450675062175e-07, 2.466972798163045e-07, 2.4537393558127494e-07, 8.620702700112414e-07, 1.3194196669345807e-09, 5.904492652622623e-09, 3.4553994366211173e-13]]
    # gd_eps_test = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [20, 6, 6, 36, 36, 36, 3, 3, 3, 3, 3, 3, 3, 3, 3], [1.97068568802946e-05, 2.327936863777599e-06, 4.451843947909425e-07, 2.396973756410826e-05, 2.6334322053867958e-05, 2.770784173437725e-05, 1.7554788893405403e-05, 1.1677548339730388e-08, 2.593894134924519e-06, 1.4325390264757815e-06, 1.3845917938177057e-06, 1.3803413848006244e-06, 1.3790148734085724e-06, 1.379048340538573e-06, 1.3790317758982276e-06]]
    # criteria_eps_test = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12, 13, 14], [0.0008837105529917022, 1.9828627352953356e-05, 4.451843947909425e-07, 4.451843947909425e-07, 9.880503880497842e-09, 2.1960668106334396e-10, 2.1960668106334396e-10, 4.7142439404436104e-12, 1.0158373324248673e-13, 1.0158373324248673e-13, 1.99463062128512e-15, 3.9568624630724025e-17, 3.9568624630724025e-17, 6.025386183451288e-19, 9.455774567734432e-21]]
    # gd_grad_eps_test = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [200, 3, 16, 6, 48, 56, 62, 5, 4, 5, 6, 6, 7, 6, 5], [1.7916455962127376, 2.225465862301394e-05, 3.026893463477357e-05, 2.3389657061528475e-07, 1.8417181170821742e-07, 8.504115840899e-09, 8.075739931476811e-10, 1.02691213163061e-11, 2.275634386312758e-13, 2.2351155756085124e-13, 1.498404347852592e-17, 1.263432179110045e-17, 1.1823701342117995e-20, 1.9469909824570053e-19, 1.4805580321621784e-23]]

    print()
    set_graphic(axes[0], grad_eps_test, "Графік 1. Залежність кількості ітерацій від точності обчисленнях похідних")
    set_graphic(axes[1], gd_eps_test, "Графік 2. Залежність кількості ітерацій від точності одновимірного пошуку")
    set_graphic(axes[2], criteria_eps_test, "Графік 3. Залежність кількості ітерацій від точності критерія зупинки")
    set_graphic(axes[3], gd_grad_eps_test, "Графік 4. Залежність кількості ітерацій від точності обчисленнях похідних та МОП")

    plt.subplots_adjust(hspace=0.5)

    fig.show()
    fig.savefig("graphics/task1.png")


####

task1()
