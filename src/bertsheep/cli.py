import argparse

from bertsheep.data import DUMP_PATH, Data
from bertsheep.experiment import (
    ARMS,
    MUTATION,
    N_REPEATS,
    Experiment,
    results_path,
)
from bertsheep.results import Results
from bertsheep.splitters import DISTRIBUTIONS

# The entry point has a module of its own because results.py imports from
# experiment.py, so neither can import the other to run both stages.


def main() -> None:
    """
    Command-line entry point, `uv run bertsheep <target> --train --results`.

    --train preprocesses the target, tunes any (arm, distribution) whose study
    is short of its trials, then runs every missing cell of the matrix. Both
    stages resume, so the same command restarts a run that died part way; the
    narrowing flags are for a smoke run whose rows then count towards the full
    grid, and they narrow the tuning too, since each arm reads only its own
    study.

    --results prints the pairwise arm tests and draws the Question 1 and 2
    figures from the target's results file. Training runs first when both are
    given, so the figures include the rows it adds.
    """
    parser = argparse.ArgumentParser(prog="bertsheep", description=main.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="short target name, a key of data.TARGET")
    parser.add_argument("--mutation", default=MUTATION,
                        help=f"variant to keep (default: {MUTATION})")
    parser.add_argument("--train", action="store_true",
                        help="tune, then run the arm x distribution x seed grid")
    parser.add_argument("--results", action="store_true",
                        help="print the arm comparisons and draw the figures")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS,
                        help="arms to train (--train only)")
    parser.add_argument("--distributions", nargs="+", choices=DISTRIBUTIONS,
                        default=DISTRIBUTIONS,
                        help="distributions to train (--train only)")
    parser.add_argument("--seeds", type=int, default=N_REPEATS,
                        help=f"train seeds 0..N-1 (default: {N_REPEATS}; --train only)")
    args = parser.parse_args()
    if not (args.train or args.results):
        parser.error("pass --train, --results or both")

    if args.train:
        df = Data(DUMP_PATH, args.target, args.mutation)._preprocess()
        exp = Experiment(df, args.target, args.mutation)
        exp.tune(args.arms, args.distributions)
        exp.grid(args.arms, args.distributions, range(args.seeds))
        print(f"-- Results in {exp.results_path}")
    if args.results:
        results = Results(results_path(args.target, args.mutation))
        print(results.comparisons())
        print(results.q1())
        print(*results.q2(), sep="\n")
