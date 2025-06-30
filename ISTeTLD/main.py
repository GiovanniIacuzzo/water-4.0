import argparse
import os

def run_train():
    from train import train
    train()

def run_test():
    from test import generate_and_evaluate
    generate_and_evaluate()

def run_preprocess():
    from dataset import preprocess_and_save_dataset
    preprocess_and_save_dataset()

def run_visualization():
    from visualizer import plot_generated_distributions
    plot_generated_distributions()

def main():
    parser = argparse.ArgumentParser(description="💧 Leakage Scenario Generator - CGAN based")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "test", "preprocess", "visualize"],
                        help="Modalità di esecuzione del main")

    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "test":
        run_test()
    elif args.mode == "preprocess":
        run_preprocess()
    elif args.mode == "visualize":
        run_visualization()
    else:
        raise ValueError("⚠️ Modalità non riconosciuta")

if __name__ == "__main__":
    main()
