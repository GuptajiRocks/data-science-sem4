n1 = [4,9,5]
n2 = [9,4,9,8,4]

nn1 = set(n1)
nn2 = set(n2)

nn1 = list(nn1)
nn2 = list(nn2)

l = 0

if (len(nn1) < len(nn2)):
    l = len(nn2)
else:
    l = len(nn1)

fin = []
for i in range(l):
    if ((nn1[i] in nn2) and (nn1[i] not in fin)):
        fin.append(nn1[i])

print(fin)