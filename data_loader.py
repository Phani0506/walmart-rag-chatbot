# data_loader.py (Updated for local project file)
import os
import pandas as pd

def load_and_prepare_data():
    print("Loading local dataset...")
    file_path = "Walmart_Sales.csv" # Looks for the file in the same folder

    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except FileNotFoundError:
        print(f"ERROR: File not found at '{file_path}'")
        print("Please make sure 'Walmart_Sales.csv' is in the same folder as your scripts.")
        return []

    print("Dataset loaded successfully. Preparing text documents...")
    df.columns = df.columns.astype(str)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    documents = [
        (
            f"On {row['Date'].strftime('%Y-%m-%d')}, at Store {row['Store']}, "
            f"weekly sales were ${row['Weekly_Sales']:.2f}. "
            f"The holiday status was {row['Holiday_Flag']} (1 for holiday, 0 for non-holiday). "
            f"The temperature was {row['Temperature']:.2f}, "
            f"fuel price was ${row['Fuel_Price']:.3f}, "
            f"CPI was {row['CPI']:.3f}, "
            f"and unemployment rate was {row['Unemployment']:.3f}."
        )
        for index, row in df.iterrows()
    ]
    print(f"✓ Converted {len(documents)} rows into text documents.")
    return documents