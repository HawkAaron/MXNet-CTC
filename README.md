# MXNet binding for Warp-ctc

> Follow [here](https://github.com/HawkAaron/mxnet-transducer#install-and-test) to build this binding in MXNet source.

Since MXNet implementation of CTC loss cannot do well with large vocab, this repo provides a binding for MXNet.


# Python wrap 

```python
class CTCLoss(gluon.loss.Loss):
    def __init__(self, layout='NTC', label_layout='NT', weight=None, **kwargs):
        assert layout in ['NTC', 'TNC'],\
            "Only 'NTC' and 'TNC' layouts for pred are supported. Got: %s"%layout
        assert label_layout in ['NT', 'TN'],\
            "Only 'NT' and 'TN' layouts for label are supported. Got: %s"%label_layout
        self._layout = layout
        self._label_layout = label_layout
        batch_axis = label_layout.find('N')
        super(CTCLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label,
                pred_lengths=None, label_lengths=None):
        if self._layout == 'NTC':
            pred = F.swapaxes(pred, 0, 1)
        if self._batch_axis == 1:
            label = F.swapaxes(label, 0, 1)

        loss = F.contrib.warpctc_loss(pred, label, pred_lengths, label_lengths)
        return loss
```
