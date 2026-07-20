def save_predictions(df, output_file):
    """
    Save combined predictions to CSV.
    """

    df.to_csv(output_file, index=False) # writes the DataFrame to disk

    print(f"\nSaved {len(df)} state vectors") # shows how many rows were written

    print(f"CSV written to:\n{output_file}") # displays the save location