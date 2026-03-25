from data_generation import generate_data
from demand_model import train_models
from optimization import run_optimization

def main():
    df = generate_data()
    models = train_models(df)
    run_optimization(models)

if __name__ == "__main__":
    main()