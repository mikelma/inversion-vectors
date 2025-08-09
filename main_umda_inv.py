from pypermu import problems
import transformations as T
from argparse import ArgumentParser
import numpy as np
from umda import Umda
from tqdm import tqdm
import pandas as pd
import os


def parse_args():
    parser = ArgumentParser()

    # fmt: off
    parser.add_argument("--instance", type=str, required=True,
        help="Path to the instance of the problem")
    parser.add_argument("--pop-size", type=int, required=False, default=40,
        help="Size of the population")
    parser.add_argument("--iters", type=int, required=False, default=100)
    parser.add_argument("--repes", type=int, required=False, default=10)

    # inversion vector options
    parser.add_argument("--v-left", type=lambda v: v == "True", default=True)
    parser.add_argument("--v-permu", type=lambda v: v == "True", default=True)
    parser.add_argument("--v-gt", type=lambda v: v == "True", default=True)

    parser.add_argument("--progress", action="store_true",
        help="If provided, a progress bar is shown")

    parser.add_argument("--csv-path", type=str, default=None,
        help="Path to the CSV file to save. If not provided, the filename is generated automatically.")

    # fmt: on
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # determine the type of problem from the instance's name
    print(args.instance.split(".")[-1])
    extension = args.instance.split(".")[-1]
    if extension == "fsp":
        problem_type = "pfsp"
        minimize = True
        problem = problems.pfsp.Pfsp(args.instance)

    elif extension == "dat":
        problem_type = "qap"
        minimize = True
        problem = problems.qap.Qap(args.instance)
    else:
        problem_type = "lop"
        minimize = False
        problem = problems.lop.Lop(args.instance)

    n = problem.size

    print(f"\n==> Probem: {problem_type}, Size: {n} <==\n")

    transf = lambda p: T.inversion(
        p, inv=not (args.v_permu), left=args.v_left, gt=args.v_gt
    )
    rev_transf = lambda v: T.rev_inversion(
        v, inv=not (args.v_permu), left=args.v_left, gt=args.v_gt
    )
    print(
        f"~~> Transformation: permu={args.v_permu}, left={args.v_left}, gt={args.v_gt}"
    )

    # fitnesses = [list() for _ in range(args.iters)]
    best_fs = [list() for _ in range(args.iters)]
    avg_fs = [list() for _ in range(args.iters)]
    repes = tqdm(range(args.repes)) if args.progress else range(args.repes)
    for repe in repes:
        # initial population
        pop = [np.random.permutation(n) for _ in range(args.pop_size)]
        umda = Umda()
        best_f = float("inf") if minimize else float("-inf")
        for it in range(args.iters):
            # evaluation
            fitness = problem.evaluate(pop)

            # selection
            bests = np.argsort(fitness)
            if not minimize:
                bests = bests[::-1]
            selected = [pop[i] for i in bests[: (args.pop_size // 2)]]

            if (minimize and best_f > np.min(fitness)) or (not minimize and best_f < np.max(fitness)):
                best_f = fitness[bests[0]]
            best_fs[it].append(best_f)
            avg_fs[it].append(np.mean(fitness))

            # learn
            pop_t = [transf(p) for p in selected]
            umda.learn(pop_t)

            # sample
            pop = [rev_transf(umda.sample()) for _ in range(args.pop_size)]

            # fitnesses[it] += list(fitness)

    df = pd.DataFrame()
    # df["fitnesses"] = np.array(fitnesses).ravel()
    # df["iteration"] = np.repeat(np.arange(args.iters), args.repes * args.pop_size)
    # df["best fitness"] = np.repeat(np.array(best_fs).ravel(), args.pop_size)
    df["iteration"] = np.repeat(np.arange(args.iters), args.repes)
    df["best fitness"] = np.array(best_fs).ravel()
    df["avg fitness"] = np.array(avg_fs).ravel()
    df["v permu"] = args.v_permu
    df["v left"] = args.v_left
    df["v gt"] = args.v_gt
    df["instance"] = args.instance.split("/")[-1]
    df["problem"] = problem_type

    fname = f"data_{args.instance.split('/')[-1]}.csv" if args.csv_path is None else args.csv_path
    print(f"Writting data to {fname}...")
    if os.path.exists(fname):
        print("  * Appending...")
        df.to_csv(fname, mode="a", header=False)
    else:
        df.to_csv(fname)
