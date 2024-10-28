package org.jesperancinha.organic

import org.jesperancinha.organic.data.generateSyntheticData
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.activation.Activations.Linear
import org.jetbrains.kotlinx.dl.api.core.activation.Activations.Relu
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.regularization.Dropout
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metric
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics.MAE
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.junit.jupiter.api.Test
import kotlin.random.Random

class OrganicAppOverfittingCorrectionTest {

    @Test
    fun `should run an overfitting test`() {
        val (trainData, testData) = generateSyntheticData()
        val model = Sequential.of(
            Input(1),
            Dense(128, activation = Activations.Relu),
            Dropout(0.3f),
            Dense(64, activation = Activations.Relu),
            Dropout(0.3f),
            Dense(64, activation = Activations.Relu),
            Dropout(0.3f),
            Dense(32, activation = Activations.Relu),
            Dropout(0.3f),
            Dense(16, activation = Activations.Relu),
            Dropout(0.3f),
            Dense(1, activation = Activations.Linear)
        )

        model.compile(
            optimizer = Adam(learningRate = 0.001f),
            loss = Losses.MSE,
            metric = MAE
        )
        var bestValidationLoss = Double.MAX_VALUE
        var epochsWithoutImprovement = 0
        val patience = 10
        val trainLosses = mutableListOf<Double>()
        val trainMAEs = mutableListOf<Double>()
        for (epoch in 1..5000) {
            model.fit(trainData, epochs = 1, batchSize = 32)
            val trainEvaluation = model.evaluate(trainData)
            trainLosses.add(trainEvaluation.lossValue)
            trainMAEs.add(trainEvaluation.metrics[MAE] ?: 0.0)
            val testEvaluation = model.evaluate(testData)
            println("Epoch $epoch: Training Loss: ${trainLosses.last()}, Training MAE: ${trainMAEs.last()}")
            println("Test Loss: ${testEvaluation.lossValue}, Test MAE: ${testEvaluation.metrics[MAE]}")
            if (testEvaluation.lossValue < bestValidationLoss) {
                bestValidationLoss = testEvaluation.lossValue
                epochsWithoutImprovement = 0 // Reset counter
                println("Validation loss improved, saving model weights.")
            } else {
                epochsWithoutImprovement++
                if (epochsWithoutImprovement >= patience) {
                    println("Early stopping triggered after $epoch epochs.")
                    break
                }
            }
        }

        val xTest = testData.x
        val predictions = xTest.map { model.predictSoftly(it) }
        val yTest = testData.y

        println("Predictions vs. True Values:")
        predictions.forEach { println(it[0].toString()) }
        for (i in predictions.indices) {
            println("True: ${yTest[i]}, Predicted: ${predictions[i][0]}")
        }
    }

}
