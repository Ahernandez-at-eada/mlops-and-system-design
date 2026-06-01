from session_4.src.source import load_data
from session_4.src.transform import Transformer, split_features_target
from session_4.src.train import train_model
from session_4.src.store import save_model


def main():
    """Run the full ML pipeline: load, transform, train, save."""
    print("Starting ML pipeline...")

    # 1. Load data
    print("\n[1/4] Loading data...")
    df = load_data()
    print(f"Data loaded. Shape: {df.shape}")

    # 2. Transform data
    print("\n[2/4] Transforming data...")
    transformer = Transformer()
    df_transformed = transformer.transform(df)
    X, y = split_features_target(df_transformed)
    print(f"Data transformed. X shape: {X.shape}, y shape: {y.shape}")

    # 3. Train model
    print("\n[3/4] Training model...")
    model = train_model(X, y)

    # 4. Save model
    print("\n[4/4] Saving model...")
    model_path = save_model(model)

    print(f"\nPipeline completed successfully. Model saved at: {model_path}")


if __name__ == "__main__":
    main()
