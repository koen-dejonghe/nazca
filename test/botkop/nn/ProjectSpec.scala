package botkop.nn

import botkop.nn.costs.Softmax
import botkop.nn.gates.{Linear, Relu}
import botkop.nn.network.NetworkConfig
import botkop.nn.optimizers.Nesterov
import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{FlatSpec, Matchers}
import play.api.libs.json.Json

class ProjectSpec extends FlatSpec with Matchers with LazyLogging {

  it should "emit correct json for mnist" in {
    val template: NetworkConfig = ((Linear + Relu) * 2)
      .withDimensions(784, 50, 10)
      .withOptimizer(Nesterov)
      .withCostFunction(Softmax)
      .withRegularization(1e-8)
      .withLearningRate(0.4)
      .withLearningRateDecay(0.99)
      .configure

    val project = Project(name = "mnist-sample-project",
      miniBatchSize = 64,
      dataSet = "mnist",
      persistenceFrequency = 30,
      template = template)

    val json = Json.prettyPrint(Json.toJson(project))

    logger.debug(json)

  }

}
