import sys
index = 0
labels = {}

def head(tokens):
    global index
    token = tokens[index]
    index += 1
    return token

def label(token):
    global labels
    if token not in labels:
        labels[token] = len(labels)
    return labels[token]

def main():
    global index
    graphs = 0
    infile = open(sys.argv[1], "r")
    outfile = open(sys.argv[2], "w")
    tokens = infile.read().strip().split()
    while index < len(tokens):
        graphs += 1
        outfile.write(f"t # {head(tokens)[1:]}\n")
        n = int(head(tokens))
        for j in range(n):
            outfile.write(f"v {j} {label(head(tokens))}\n")
        m = int(head(tokens))
        for j in range(m):
            id = "u" if sys.argv[3] == "fsg" else "e"
            outfile.write(f"{id} {head(tokens)} {head(tokens)} {head(tokens)}\n")
    outfile.close()
    infile.close()
    sys.exit(str(graphs))

main()
