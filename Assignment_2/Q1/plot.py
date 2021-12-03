import matplotlib.pyplot as plt
import os, sys, time

path = os.path.dirname(sys.argv[0])
supports = [5, 10, 25, 50, 95]
gspan = []
fsg = []
gaston = []

for support in supports:
    t = time.time()
    os.system(f"timeout 1h {path}/gSpan6/gSpan-64 -s {support/100} -f {sys.argv[1]} -o -i")
    gspan.append((time.time()-t)/60)
    t = time.time()
    os.system(f"timeout 1h {path}/pafi-1.0.1/Linux/fsg -s {support} {sys.argv[2]}")
    fsg.append((time.time()-t)/60)
    t = time.time()
    os.system(f"timeout 1h {path}/gaston-1.1/gaston {(int(sys.argv[4])*support)/100} {sys.argv[3]} {sys.argv[3]}.fp")
    gaston.append((time.time()-t)/60)

barWidth = 0.25
fig = plt.subplots(figsize =(10, 8))

br1 = [1, 2, 3, 4, 5]
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

plt.bar(br1, gspan, color = 'y', width = barWidth, edgecolor = 'grey', label = 'gSpan')
plt.bar(br2, fsg, color = 'g', width = barWidth, edgecolor = 'grey', label = 'FSG') 
plt.bar(br3, gaston, color = 'c', width = barWidth, edgecolor = 'grey', label = 'Gaston') 

plt.title('Performance Graph')
plt.ylabel('Running Time (Minutes)')
plt.xlabel('Minimum Support')
plt.xticks([x for x in br2], [str(support) + "%" for support in supports])

plt.legend()
plt.savefig(f"{sys.argv[5]}.png")
