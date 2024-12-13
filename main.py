# Main Function
def main():
    # File path to the dataset (update this to your dataset path)
    file_path = "Ames.csv"

    # Load and preprocess the dataset
    print("Loading and preprocessing data...")
    df = load_data(file_path)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df, target_column='Label')

    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create the model
    print("Creating the model...")
    input_dim = X_train.shape[1]
    model = MutagenicActivityModel(input_dim)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, val_loader, criterion, optimizer, device)

    # Load the best model and evaluate
    print("Evaluating the model...")
    model.load_state_dict(torch.load("best_model.pth"))
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
