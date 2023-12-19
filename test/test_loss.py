# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from modulus.sym.loss import (
    PointwiseLossNorm,
    DecayedPointwiseLossNorm,
    IntegralLossNorm,
    DecayedIntegralLossNorm,
)


def test_loss_norm():
    invar = {
        "x": paddle.arange(end=10)[:, None],
        "area": paddle.ones(shape=[10])[:, None] / 10,
    }
    pred_outvar = {"u": paddle.arange(end=10)[:, None]}
    true_outvar = {"u": paddle.arange(end=10)[:, None] + 2}
    lambda_weighting = {"u": paddle.ones(shape=[10])[:, None]}
    loss = PointwiseLossNorm(2)
    l = loss.forward(invar, pred_outvar, true_outvar, lambda_weighting, step=0)
    assert paddle.isclose(x=l["u"], y=paddle.to_tensor(data=4.0))
    loss = PointwiseLossNorm(1)
    l = loss.forward(invar, pred_outvar, true_outvar, lambda_weighting, step=0)
    assert paddle.isclose(x=l["u"], y=paddle.to_tensor(data=2.0))
    loss = DecayedPointwiseLossNorm(2, 1, decay_steps=1000, decay_rate=0.5)
    l = loss.forward(invar, pred_outvar, true_outvar, lambda_weighting, step=0)
    assert paddle.isclose(x=l["u"], y=paddle.to_tensor(data=4.0))
    l = loss.forward(invar, pred_outvar, true_outvar, lambda_weighting, step=1000)
    assert paddle.isclose(x=l["u"], y=paddle.to_tensor(data=2.82842712))
    l = loss.forward(invar, pred_outvar, true_outvar, lambda_weighting, step=1000000)
    assert paddle.isclose(x=l["u"], y=paddle.to_tensor(data=2.0))
    list_invar = [
        {
            "x": paddle.arange(end=10)[:, None],
            "area": paddle.ones(shape=[10])[:, None] / 10,
        }
    ]
    list_pred_outvar = [{"u": paddle.arange(end=10)[:, None]}]
    list_true_outvar = [{"u": paddle.to_tensor(data=2.5)[None, None]}]
    list_lambda_weighting = [{"u": paddle.ones(shape=[1])[None, None]}]
    loss = IntegralLossNorm(2)
    l = loss.forward(
        list_invar, list_pred_outvar, list_true_outvar, list_lambda_weighting, step=0
    )
    assert paddle.isclose(x=l["u"], y=paddle.to_tensor(data=4.0))
    loss = IntegralLossNorm(1)
    l = loss.forward(
        list_invar, list_pred_outvar, list_true_outvar, list_lambda_weighting, step=0
    )
    assert paddle.isclose(x=l["u"], y=paddle.to_tensor(data=2.0))
    loss = DecayedIntegralLossNorm(2, 1, decay_steps=1000, decay_rate=0.5)
    l = loss.forward(
        list_invar, list_pred_outvar, list_true_outvar, list_lambda_weighting, step=0
    )
    assert paddle.isclose(x=l["u"], y=paddle.to_tensor(data=4.0))
    l = loss.forward(
        list_invar, list_pred_outvar, list_true_outvar, list_lambda_weighting, step=1000
    )
    assert paddle.isclose(x=l["u"], y=paddle.to_tensor(data=2.82842712))
    l = loss.forward(
        list_invar,
        list_pred_outvar,
        list_true_outvar,
        list_lambda_weighting,
        step=1000000,
    )
    assert paddle.isclose(x=l["u"], y=paddle.to_tensor(data=2.0))
