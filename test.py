import glob

jvocab = {}

jlines = open('jp.txt').read().split('\n')
for i in range(len(jlines)):
    lt = jlines[i].split()
    for w in lt:
        if w not in jvocab:
            jvocab[w] = len(jvocab)
jvocab['<eos>'] = len(jvocab)
jv = len(jvocab)
print(jlines)
print(jvocab)
print(jv)

evocab = {}

elines = glob.glob('python_output/*.py')
for i in range(len(elines)):
    f = open(elines[i], 'r').read().split('\n')
    for line in f:
        if line not in evocab:
            evocab[line] = len(evocab)
evocab["<eos>"] = len(evocab)
ev = len(evocab)
print(elines)
print(evocab)
print(ev)
