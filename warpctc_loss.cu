/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file warpctc_loss.cc
 * \brief
 * \author Mingkun Huang
*/

#include "./warpctc_loss-inl.h"
#include "./warpctc_include/detail/gpu_ctc.h"

namespace mshadow {

template <typename DType>
ctcStatus_t compute_ctc_cost(const Tensor<gpu, 3, DType> activations,
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

    int minibatch = static_cast<int>(activations.size(1));
    int alphabet_size = static_cast<int>(activations.size(2));
    baidu_warpctc::GpuCTC<DType> ctc(alphabet_size, minibatch, workspace,
                    activations.stream_->stream_, blank_label);
    if (train)
        return ctc.cost_and_grad(activations.dptr_, gradients, costs, 
                                flat_labels, label_lengths, input_lengths);
    else
        return ctc.score_forward(activations.dptr_, costs, flat_labels,
                                label_lengths, input_lengths);
}

}  // namespace mshadow

////////////////////////////////////////////////////////////////////////////////

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<gpu>(WARPCTCLossParam param, int dtype) {
    return new WARPCTCLossOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
