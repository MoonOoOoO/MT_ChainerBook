import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

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
id2wd = {}
elines = open('eng.txt').read().split('\n')
# allocate an ID to each English word, match the IDs to the English words and saved in array "id2wd"
for i in range(len(elines)):
    lt = elines[i].split()
    for w in lt:
        if w not in evocab:
            val = len(evocab)
            evocab[w] = val
            id2wd[val] = w
val = len(evocab)
evocab['<eos>'] = val
id2wd[val] = '<eos>'
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
        self.H.reset_state()
        for i in range(len(jline)):
            wid = jvocab[jline[i]]
            x_k = self.embedx(Variable(np.array([wid], dtype=np.int32)))
            h = self.H(x_k)
        x_k = self.embedx(Variable(np.array([jvocab['<eos>']], dtype=np.int32)))
        tx = Variable(np.array([evocab[eline[0]]], dtype=np.int32))
        h = self.H(x_k)
        accum_loss = F.softmax_cross_entropy(self.W(h), tx)
        for i in range(1, len(eline)):
            wid = evocab[eline[i]]
            x_k = self.embedy(Variable(np.array([wid], dtype=np.int32)))
            next_wid = evocab['<eos>'] if (i == len(eline) - 1) else evocab[eline[i + 1]]
            tx = Variable(np.array([next_wid], dtype=np.int32))
            h = self.H(x_k)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss


demb = 100
model = MyMT(jv, ev, demb)
# cuda.get_device(0).use()
# model.to_gpu()
optimizer = optimizers.Adam()
optimizer.setup(model)
for epoch in range(100):
    for i in range(len(jlines) - 1):
        jln = jlines[i].split()
        jlnr = jln[::-1]
        eln = elines[i].split()
        model.H.reset_state()
        model.zerograds()
        loss = model(jlnr, eln)
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        print(i, "finished")
    outfile = "model/mt-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)
    print("mt-" + str(epoch) + ".model")
