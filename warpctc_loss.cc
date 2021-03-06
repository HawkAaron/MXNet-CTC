#include "./warpctc_loss-inl.h"
#include "./warpctc_include/detail/cpu_ctc.h"

namespace mshadow {

template<typename DType>
ctcStatus_t compute_ctc_cost(const Tensor<cpu, 3, DType> activations,
                             DType* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             DType *costs,
                             void *workspace,
                             int train,
                             int blank_label) {

    if (flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        workspace == nullptr)
        return CTC_STATUS_INVALID_VALUE;

    int minibatch = activations.size(1);
    int alphabet_size = activations.size(2);
    baidu_warpctc::CpuCTC<DType> ctc(alphabet_size, minibatch, workspace, 0, blank_label);

    if (train)
        return ctc.cost_and_grad(activations.dptr_, gradients,
                                    costs,
                                    flat_labels, label_lengths,
                                    input_lengths);
    else
        return ctc.score_forward(activations.dptr_, costs, flat_labels,
                                    label_lengths, input_lengths);
}

}

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<cpu>(WARPCTCLossParam param, int dtype) {
  return new WARPCTCLossOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *WARPCTCLossProp::CreateOperatorEx(Context ctx,
                                        std::vector<TShape> *in_shape,
                                        std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(WARPCTCLossParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_WARPCTCLoss, WARPCTCLossProp)
    .describe(R"code(Connectionist Temporal Classification Loss.
The shapes of the inputs and outputs:
- **data**: `(sequence_length, batch_size, alphabet_size)`
- **label**: `(batch_size, label_sequence_length)`
- **out**: `(batch_size)`
The `data` tensor consists of sequences of activation vectors (without applying softmax),
with i-th channel in the last dimension corresponding to i-th label
for i between 0 and alphabet_size-1 (i.e always 0-indexed).
Alphabet size should include one additional value reserved for blank label.
When `blank_label` is ``"first"``, the ``0``-th channel is be reserved for
activation of blank label, or otherwise if it is "last", ``(alphabet_size-1)``-th channel should be
reserved for blank label.
``label`` is an index matrix of integers. When `blank_label` is ``"first"``,
the value 0 is then reserved for blank label, and should not be passed in this matrix. Otherwise,
when `blank_label` is ``"last"``, the value `(alphabet_size-1)` is reserved for blank label.
If a sequence of labels is shorter than *label_sequence_length*, use the special
padding value at the end of the sequence to conform it to the correct
length. The padding value is `0` when `blank_label` is ``"first"``, and `-1` otherwise.
For example, suppose the vocabulary is `[a, b, c]`, and in one batch we have three sequences
'ba', 'cbb', and 'abac'. When `blank_label` is ``"first"``, we can index the labels as
`{'a': 1, 'b': 2, 'c': 3}`, and we reserve the 0-th channel for blank label in data tensor.
The resulting `label` tensor should be padded to be::
  [[2, 1, 0, 0], [3, 2, 2, 0], [1, 2, 1, 3]]
When `blank_label` is ``"last"``, we can index the labels as
`{'a': 0, 'b': 1, 'c': 2}`, and we reserve the channel index 3 for blank label in data tensor.
The resulting `label` tensor should be padded to be::
  [[1, 0, -1, -1], [2, 1, 1, -1], [0, 1, 0, 2]]
``out`` is a list of WARPCTC loss values, one per example in the batch.
See *Connectionist Temporal Classification: Labelling Unsegmented
Sequence Data with Recurrent Neural Networks*, A. Graves *et al*. for more
information on the definition and the algorithm.
)code" ADD_FILELINE)
    .add_argument("data", "NDArray-or-Symbol", "Input data to the warpctc_loss op.")
    .add_argument("label", "NDArray-or-Symbol",
                  "Ground-truth labels for the loss.")
    .add_argument("data_lengths", "NDArray-or-Symbol",
                  "Lengths of data for each of the samples. Only required "
                  "when use_data_lengths is true.")
    .add_argument("label_lengths", "NDArray-or-Symbol",
                  "Lengths of labels for each of the samples. Only required "
                  "when use_label_lengths is true.")
    .add_arguments(WARPCTCLossParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_WARPCTCLoss).add_alias("_contrib_warpctc_loss");

}  // namespace op
}  // namespace mxnet
