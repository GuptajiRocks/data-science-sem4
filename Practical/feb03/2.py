import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("Practical//feb03//rd.csv")

transactions = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

te = TransactionEncoder()
encoded_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(encoded_array, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.6)

print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules)

plt.scatter(rules["support"], rules["confidence"])
plt.show()
