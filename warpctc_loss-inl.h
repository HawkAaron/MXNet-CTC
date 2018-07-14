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
 * \file warpctc_loss-inl.h
 * \brief
 * \author Mingkun Huang
*/

#ifndef MXNET_OPERATOR_CONTRIB_WARPCTC_LOSS_INL_H_
#define MXNET_OPERATOR_CONTRIB_WARPCTC_LOSS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "../operator_common.h"
#include "../sequence_op_common.h"
#include "../mshadow_op.h"
#include "../nn/sequence_mask-inl.h"

namespace mxnet {
namespace op {

namespace warpctc_loss {
enum WARPCTCLossOpInputs { kData, kLabel, kInputLength, kLabelLength };
enum WARPCTCLossOpOutputs { kOut, kGrad };
enum WARPCTCLossOpForwardResource { kTempSpace };
}

template <typename T>
inline void get_workspace_size(const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size, int minibatch,
                               bool gpu,
                               size_t* size_bytes)
{
    if (label_lengths == nullptr ||
        input_lengths == nullptr ||
        size_bytes == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return ;

    // This is the max of all S and T for all examples in the minibatch.
    int maxL = *std::max_element(label_lengths, label_lengths + minibatch);
    int maxT = *std::max_element(input_lengths, input_lengths + minibatch);

    const int S = 2 * maxL + 1;

    *size_bytes = 0;

    if (gpu) {
        // GPU storage
        //nll_forward, nll_backward
        *size_bytes += 2 * sizeof(float) * minibatch;

        //repeats
        *size_bytes += sizeof(int) * minibatch;

        //label offsets
        *size_bytes += sizeof(int) * minibatch;

        //utt_length
        *size_bytes += sizeof(int) * minibatch;

        //label lengths
        *size_bytes += sizeof(int) * minibatch;

        //labels without blanks - overallocate for now
        *size_bytes += sizeof(int) * maxL * minibatch;

        //labels with blanks
        *size_bytes += sizeof(int) * S * minibatch;

        //alphas
        *size_bytes += sizeof(float) * S * maxT * minibatch;

        //denoms
        *size_bytes += sizeof(float) * maxT * minibatch;

        //probs (since we will pass in activations)
        *size_bytes += sizeof(float) * alphabet_size * maxT * minibatch;

    } else {
        //cpu can eventually replace all minibatch with
        //max number of concurrent threads if memory is
        //really tight

        //per minibatch memory
        size_t per_minibatch_bytes = 0;

        //output
        per_minibatch_bytes += sizeof(float) * alphabet_size ;

        //alphas
        per_minibatch_bytes += sizeof(float) * S * maxT;

        //betas
        per_minibatch_bytes += sizeof(float) * S;

        //labels w/blanks, e_inc, s_inc
        per_minibatch_bytes += 3 * sizeof(int) * S;

        *size_bytes = per_minibatch_bytes * minibatch;

        //probs
        *size_bytes += sizeof(float) * alphabet_size * maxT * minibatch;
    }
}

// Takes a tensor of labels, and a vector which specifies the actual length of each label
// The tensor is packed into an std::vector without padding characters.
// The label length vector is copied into an std::vector.
// When cudnn is enabled, the return value signifies whether the cudnn length limit is exceeded.
template <typename xpu>
inline bool PackLabelByLength(mshadow::Tensor<xpu, 2, int32_t> labels,
                              mshadow::Tensor<xpu, 1, int32_t> in_label_lengths,
                              std::vector<int> *packed_labels,
                              std::vector<int> *label_lengths) {
  int batch = labels.size(0);
  int max_num_labels = labels.size(1);
  bool exceed_limit = false;

  IndexTensorToVector(in_label_lengths, label_lengths);

  std::vector<int> cpu_labels(max_num_labels*batch);
  mshadow::Tensor<xpu, 1, int32_t> flat_labels = labels.FlatTo1D();
  IndexTensorToVector(flat_labels, &cpu_labels);

  for (int b = 0; b < batch; ++b) {
    auto start = cpu_labels.data()+b*max_num_labels;
    int len = label_lengths->at(b);
// #if defined(__CUDACC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
//     exceed_limit = exceed_limit || len > CUDNN_LABEL_LENGTH_LIMIT;
// #endif
    std::copy(start, start + len,
              std::back_inserter(*packed_labels));
  }
  return exceed_limit;
}

struct WARPCTCLossParam : public dmlc::Parameter<WARPCTCLossParam> {
  int blank_label;
  DMLC_DECLARE_PARAMETER(WARPCTCLossParam) {
    DMLC_DECLARE_FIELD(blank_label)
      .set_default(0)
      .describe("Set the label that is reserved for blank label.");
  }
};

template <typename xpu>
class WARPCTCLossOp : public Operator {
 public:
  explicit WARPCTCLossOp(WARPCTCLossParam p) {
    this->param_ = p;
    exceed_cudnn_limit = false;
  }

  ~WARPCTCLossOp() {

  }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 4U);
    CHECK_EQ(out_data.size(), 2U);
    exceed_cudnn_limit = false; // not use now
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 3, real_t> data =
        in_data[warpctc_loss::kData].get<xpu, 3, real_t>(s);
    Tensor<xpu, 2, int32_t> labels =
        in_data[warpctc_loss::kLabel].get<xpu, 2, int32_t>(s);
    
    Tensor<xpu, 1, real_t> costs =
        out_data[warpctc_loss::kOut].get<xpu, 1, real_t>(s);
    Tensor<xpu, 3, real_t> grad =
        out_data[warpctc_loss::kGrad].get<xpu, 3, real_t>(s);

    int max_seq_len = data.size(0);
    int batch_size = data.size(1);
    int alphabet_size = data.size(2);

    // data_lengths
    std::vector<int> data_lengths(batch_size, max_seq_len);
    IndexTensorToVector(in_data[warpctc_loss::kInputLength].get<xpu, 1, int32_t>(s), &data_lengths);

    // label_lengths
    std::vector<int> packed_labels;
    std::vector<int> label_lengths(batch_size);
    PackLabelByLength(labels, in_data[warpctc_loss::kLabelLength].get<xpu, 1, int32_t>(s), &packed_labels, &label_lengths);

    // allocate temporary workspace
    size_t size_bytes;
    bool gpu = data.kDevCPU ? false : true;

    get_workspace_size<real_t>(label_lengths.data(), data_lengths.data(),
                    alphabet_size, batch_size, gpu, &size_bytes);

    // round-up so there are enough elems in memory
    int num_tmp_elems = (size_bytes + sizeof(real_t) - 1) / sizeof(real_t);
    Tensor<xpu, 1, real_t> workspace =
        ctx.requested[warpctc_loss::kTempSpace].get_space_typed<xpu, 1, real_t>(
            Shape1(num_tmp_elems), s);

    compute_ctc_cost(data, grad.dptr_, packed_labels.data(),
                     label_lengths.data(), data_lengths.data(),
                     costs.dptr_, workspace.dptr_, 
                     req[warpctc_loss::kGrad] != mxnet::kNullOp,
                     param_.blank_label);

  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 3, real_t> data_grad =
        in_grad[warpctc_loss::kData].get<xpu, 3, real_t>(s);
    Tensor<xpu, 1, real_t> output_grad =
        out_grad[warpctc_loss::kOut].get<xpu, 1, real_t>(s);

    Tensor<xpu, 3, real_t> data_grad_computed =
        out_data[warpctc_loss::kGrad].get<xpu, 3, real_t>(s);

    Assign(data_grad, req[warpctc_loss::kData],
           mshadow::expr::broadcast<1>(output_grad, data_grad.shape_) * data_grad_computed);
  }

 private:
  WARPCTCLossParam param_;
  bool exceed_cudnn_limit;

};  // class WARPCTCLossOp

template <typename xpu>
Operator *CreateOp(WARPCTCLossParam param, int dtype);

// #if DMLC_USE_CXX11
class WARPCTCLossProp : public OperatorProperty {
 public:
  int NumVisibleOutputs() const override { return 1; }

  int NumOutputs() const override { return 2; }

  std::vector<std::string> ListArguments() const override {
    return {"data", "label", "data_lengths", "label_lengths"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "grad"};
  }

  void Init(const std::vector<std::pair<std::string, std::string>> &kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape, std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    index_t expected_inputs = 4;
    CHECK_EQ(in_shape->size(), expected_inputs)
        << "Expect " << expected_inputs << " inputs to the symbol.";

    const TShape &dshape = (*in_shape)[warpctc_loss::kData];
    const TShape &lshape = (*in_shape)[warpctc_loss::kLabel];
    CHECK_EQ(dshape.ndim(), 3U) << "The data array must be of rank 3.";
    CHECK_EQ(lshape.ndim(), 2U) << "The labels array must be of rank 2.";
    CHECK_EQ(dshape[1], lshape[0])
        << "The batch size for the labels and data arrays must be the same.";

    const TShape &dlshape = (*in_shape)[warpctc_loss::kInputLength];
    CHECK_EQ(dlshape.ndim(), 1U) << "Data length array must be a vector.";
    CHECK_EQ(dlshape[0], dshape[1])
        << "The batch size for the data and data lengths must be the same.";

    const TShape &llshape = (*in_shape)[warpctc_loss::kLabelLength];
    CHECK_EQ(llshape[0], lshape[0])
        << "The batch size for the labels and label lengths must be the same.";

    TShape oshape(1);
    oshape[0] = dshape[1];  // batch size
    out_shape->clear();
    out_shape->push_back(oshape);
    out_shape->push_back(dshape);  // grad output
    return true;
  }

  bool InferType(std::vector<int> *in_type, std::vector<int> *out_type,
                    std::vector<int> *aux_type) const override {
    // trans_acts, pred_acts, labels, input_length, label_length
    CHECK_LE(in_type->size(), this->ListArguments().size());
    int n_in = this->ListArguments().size();
    for (unsigned i = 0; i < in_type->size(); ++i) {
        auto type = mshadow::default_type_flag;
        if (i >= 1) type = mshadow::kInt32;
        CHECK(in_type->at(i) == type ||
            in_type->at(i) == -1) << "Unsupported data type " << in_type->at(i);
    }
    in_type->clear();
    for (int i = 0; i < n_in; ++i ) {
        auto type = mshadow::default_type_flag;
        if (i >= 1) type = mshadow::kInt32;
        in_type->push_back(type);
    }

    int n_out = this->ListOutputs().size();
    out_type->clear();
    for (int i = 0; i < n_out; ++i ) out_type->push_back(mshadow::default_type_flag);

    int n_aux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (int i = 0; i < n_aux; ++i ) aux_type->push_back(mshadow::default_type_flag);
    return true;
  }

  OperatorProperty *Copy() const override {
    auto ptr = new WARPCTCLossProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override { return "_contrib_WARPCTCLoss"; }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad, const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {out_grad[warpctc_loss::kOut], out_data[warpctc_loss::kGrad]};
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  WARPCTCLossParam param_;
};      // class WARPCTCLossProp
// #endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_WARPCTC_LOSS_INL_H_
