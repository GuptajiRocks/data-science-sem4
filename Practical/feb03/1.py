import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv("Practical//feb03//id.csv", sep=",", quotechar='"', header=None)

df[0] = df[0].str.strip()
df[0] = df[0].str.replace(' ', '')

df['items'] = df[0].str.split(',')

dataset = df['items'].tolist()

te = TransactionEncoder()
te_array = te.fit_transform(dataset)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.20, use_colnames=True)
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.50)
print(rules)

rules_lift = association_rules(frequent_itemsets, metric="lift", min_threshold=0.65)

disp3 = rules_lift[["antecedents","consequents","antecedent support","consequent support", "support"]]
print(disp3)

