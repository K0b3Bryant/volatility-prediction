import os
from preprocessing import csv_reader, cleaner
from volatility import meassures
from models import PredictionManager
from strategies import Strategy, StrategyManager
from performance import generate_performance_report

def main():
    # Paths
    data_path = "./data/input.csv"
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load and clean data
    print("Loading and cleaning data...")
    data = csv_reader(data_path)
    cleaned_data = cleaner(data)

    # Step 2: Generate volatility measures
    print("Generating volatility measures...")
    volatility_data = meassures(cleaned_data, high='high', low='low', close='close', open='open', window_size=3)

    # Step 3: Train models and predict
    print("Training models...")
    manager = PredictionManager()
    manager.add_model('lightgbm', use_scaler=False, n_estimators=100, learning_rate=0.1)
    manager.train(volatility_data, y=volatility_data['close'], train_index=cleaned_data.index[:-100])
    manager.predict(volatility_data, test_index=cleaned_data.index[-100:])

    # Step 4: Strategy backtesting
    print("Backtesting strategies...")
    strategy_manager = StrategyManager(risk_free_returns=volatility_data['close'])
    strategy = Strategy(
        name="Example Strategy",
        signals=volatility_data['vol_3d'],
        risky_returns=volatility_data['close'],
        risk_free_returns=volatility_data['close'] * 0  # Risk-free return example
    )
    strategy_manager.add_strategy(strategy, position_func=lambda x, **kwargs: 1)

    # Step 5: Evaluate performance
    print("Generating performance report...")
    performance_df = generate_performance_report(strategy_manager)
    performance_df.to_csv(os.path.join(output_dir, "performance.csv"), index=False)

    print("Workflow completed. Outputs saved to:", output_dir)

if __name__ == "__main__":
    main()
