package botkop.nn.gates

import botkop.nn.optimizers.Optimizer
import botkop.numsca.Tensor

case class LinearState(w: Tensor, b: Tensor, optimizer: Optimizer)
