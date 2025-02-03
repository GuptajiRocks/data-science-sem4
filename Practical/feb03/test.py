import pandas as pd
def create_one_hot_encoding(transactions):
    unique_items = sorted(list(set(item for transaction in transactions for item in transaction)))
    item_map = {item: idx for idx, item in enumerate(unique_items)}
    binary_data = []
    
    for transaction in transactions:
        row = [0] * len(unique_items)
        for item in transaction:
            row[item_map[item]] = 1
        binary_data.append(row)
    
    return binary_data, item_map

transactions = pd.read_csv("Practical//feb03//rd.csv")

binary_data, item_map = create_one_hot_encoding(transactions)
print("Item Mapping:", item_map)
print("Binary Data:")
for row in binary_data:
    print(row)