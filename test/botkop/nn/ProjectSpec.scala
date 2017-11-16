package botkop.nn

import botkop.nn.costs.Softmax
import botkop.nn.gates.{BatchNorm, Dropout, Linear, Relu}
import botkop.nn.optimizers.Nesterov
import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{FlatSpec, Matchers}
import play.api.libs.json.Json

class ProjectSpec extends FlatSpec with Matchers with LazyLogging {

  "A Project" should "emit correct json" in {

    val t1 = ((Linear + BatchNorm + Relu + Dropout) * 2 + Linear)
      .withDimensions(32 * 32 * 3, 100, 50, 10)
      .withOptimizer(Nesterov)
      .withCostFunction(Softmax)
      .withRegularization(1e-3)
      .withLearningRate(0.8)
      .configure

    val project = Project(name = "test-project",
                          miniBatchSize = 64,
                          dataSet = "cifar-10",
                          persistenceFrequency = 30,
                          template = t1)

    val json = Json.prettyPrint(Json.toJson(project))

    logger.debug(json)


    assert(json ==
      """{
        |  "name" : "test-project",
        |  "miniBatchSize" : 64,
        |  "dataSet" : "cifar-10",
        |  "persistenceFrequency" : 30,
        |  "template" : {
        |    "configs" : [ {
        |      "class" : "LinearConfig",
        |      "data" : {
        |        "shape" : [ 100, 3072 ],
        |        "regularization" : 0.001,
        |        "optimizer" : {
        |          "class" : "NesterovOptimizer",
        |          "data" : {
        |            "learningRate" : 0.8,
        |            "learningRateDecay" : 0.95,
        |            "beta" : 0.9
        |          }
        |        },
        |        "seed" : 231
        |      }
        |    }, {
        |      "class" : "BatchNormConfig",
        |      "data" : {
        |        "shape" : [ 100, 3072 ],
        |        "eps" : 0.000009999999747378752,
        |        "momentum" : 0.8999999761581421
        |      }
        |    }, {
        |      "class" : "ReluConfig"
        |    }, {
        |      "class" : "DropoutConfig",
        |      "data" : {
        |        "p" : 0.5
        |      }
        |    }, {
        |      "class" : "LinearConfig",
        |      "data" : {
        |        "shape" : [ 50, 100 ],
        |        "regularization" : 0.001,
        |        "optimizer" : {
        |          "class" : "NesterovOptimizer",
        |          "data" : {
        |            "learningRate" : 0.8,
        |            "learningRateDecay" : 0.95,
        |            "beta" : 0.9
        |          }
        |        },
        |        "seed" : 231
        |      }
        |    }, {
        |      "class" : "BatchNormConfig",
        |      "data" : {
        |        "shape" : [ 50, 100 ],
        |        "eps" : 0.000009999999747378752,
        |        "momentum" : 0.8999999761581421
        |      }
        |    }, {
        |      "class" : "ReluConfig"
        |    }, {
        |      "class" : "DropoutConfig",
        |      "data" : {
        |        "p" : 0.5
        |      }
        |    }, {
        |      "class" : "LinearConfig",
        |      "data" : {
        |        "shape" : [ 10, 50 ],
        |        "regularization" : 0.001,
        |        "optimizer" : {
        |          "class" : "NesterovOptimizer",
        |          "data" : {
        |            "learningRate" : 0.8,
        |            "learningRateDecay" : 0.95,
        |            "beta" : 0.9
        |          }
        |        },
        |        "seed" : 231
        |      }
        |    }, {
        |      "class" : "OutputConfig",
        |      "data" : {
        |        "cost" : {
        |          "class" : "Softmax"
        |        }
        |      }
        |    } ]
        |  }
        |}""".stripMargin)

  }

}
