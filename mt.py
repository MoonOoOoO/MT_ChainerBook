import numpy as xp
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

xp = cuda.cupy

jvocab = {}
jlines = open('jp.txt').read().split('\n')
# allocate an ID to each Chinese word
for i in range(len(jlines)):
    lt = jlines[i].split()
    for w in lt:
        if w not in jvocab:
            jvocab[w] = len(jvocab)
jvocab['<eos>'] = len(jvocab)
jv = len(jvocab)

evocab = {}
# id2wd = {}
elines = open('eng.txt').read().split('\n')
# allocate an ID to each English word, match the IDs to the English words and saved in array "id2wd"
for i in range(len(elines)):
    lt = elines[i].split()
    for w in lt:
        if w not in evocab:
            evocab[w] = len(evocab)
evocab['<eos>'] = len(evocab)
ev = len(evocab)


class MyMT(chainer.Chain):
    def __init__(self, jv, ev, k):
        super(MyMT, self).__init__(
            # embedx=L.EmbedID(jv, k),
            # embedy=L.EmbedID(ev, k),
            # H=L.LSTM(k, k),
            # W=L.Linear(k, ev),
        )
        with self.init_scope():
            self.embedx = L.EmbedID(jv, k)
            self.embedy = L.EmbedID(ev, k)
            self.H = L.LSTM(k, k)
            self.W = L.Linear(k, ev)

    def __call__(self, jline, eline):
        # self.H.reset_state()
        for i in range(len(jline)):
            wid = jvocab[jline[i]]
            x_k = self.embedx(Variable(xp.array([wid], dtype=xp.int32)))
            h = self.H(x_k)
        x_k = self.embedx(Variable(xp.array([jvocab['<eos>']], dtype=xp.int32)))
        tx = Variable(xp.array([evocab[eline[0]]], dtype=xp.int32))
        h = self.H(x_k)
        accum_loss = F.softmax_cross_entropy(self.W(h), tx)
        for i in range(len(eline)):
            wid = evocab[eline[i]]
            x_k = self.embedy(Variable(xp.array([wid], dtype=xp.int32)))
            next_wid = evocab['<eos>'] if (i == len(eline) - 1) else evocab[eline[i + 1]]
            tx = Variable(xp.array([next_wid], dtype=xp.int32))
            h = self.H(x_k)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            # accum_loss = loss if accum_loss is None else accum_loss + loss
            accum_loss += loss
        return accum_loss


demb = 100
model = MyMT(jv, ev, demb)
cuda.get_device_from_id(0).use()
model.to_gpu()
optimizer = optimizers.Adam()
optimizer.setup(model)
for epoch in range(100):
    for i in range(len(jlines) - 1):
        jln = jlines[i].split()
        jlnr = jln[::-1]
        eln = elines[i].split()
        model.H.reset_state()
        model.cleargrads()
        loss = model(jlnr, eln)
        loss.backward()
        loss.unchain_backward()  # truncate
        optimizer.update()
        print(i, "finished")
    outfile = "model/mt-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)
    print("mt-" + str(epoch) + ".model")
