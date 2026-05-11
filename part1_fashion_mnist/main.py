# main.py

import os

from train import main as train_main
from evaluate import main as evaluate_main
from visualize import main as visualize_main
from gui_app import main as gui_main


def display_menu():

    print("\n===================================")
    print(" Fashion-MNIST Neural Network App")
    print("===================================")

    print("1. Train Model")
    print("2. Evaluate Model")
    print("3. Visualize Results")
    print("4. Run Full Pipeline")
    print("5. Launch GUI")
    print("6. Exit")


def choose_model():

    while True:

        print("\nSelect model:")
        print("1. CNN")
        print("2. MLP")

        model_choice = input("Enter choice: ")

        if model_choice == "1":
            return "cnn"

        elif model_choice == "2":
            return "mlp"

        else:
            print("Invalid choice.")


def main():

    while True:

        display_menu()

        choice = input("\nEnter your choice: ")

        # -----------------------------
        # Train
        # -----------------------------
        if choice == "1":

            model_type = choose_model()

            print(f"\nStarting {model_type.upper()} training...\n")

            os.system(f"python train.py --model {model_type}")

        # -----------------------------
        # Evaluate
        # -----------------------------
        elif choice == "2":

            model_type = choose_model()

            print(f"\nEvaluating {model_type.upper()}...\n")

            os.system(f"python evaluate.py --model {model_type}")

        # -----------------------------
        # Visualize
        # -----------------------------
        elif choice == "3":

            print("\nGenerating plots...\n")

            os.system("python visualize.py")

        # -----------------------------
        # Full pipeline
        # -----------------------------
        elif choice == "4":

            model_type = choose_model()

            print(f"\nRunning full {model_type.upper()} pipeline...\n")

            os.system(f"python train.py --model {model_type}")
            os.system(f"python evaluate.py --model {model_type}")
            os.system("python visualize.py")

        # -----------------------------
        # GUI
        # -----------------------------
        elif choice == "5":

            print("\nLaunching GUI...\n")

            gui_main()

        # -----------------------------
        # Exit
        # -----------------------------
        elif choice == "6":

            print("\nExiting program.")
            break

        else:

            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main()