package org.jesperancinha.organic

import org.jesperancinha.organic.data.generateSyntheticData
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations.*
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses.*
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.junit.jupiter.api.Test

class OrganicAppUnderfittingCorrectionTest {

    @Test
    fun `should run a model created with corrected underfitting`() {
        val (trainData, testData) = generateSyntheticData()

        val model = Sequential.of(
            Input(1),
            Dense(256, activation = Relu),
            Dense(128, activation = Relu),
            Dense(64, activation = Relu),
            Dense(32, activation = Relu),
            Dense(16, activation = Relu),
            Dense(1, activation = Linear)
        )

        model.compile(
            optimizer = Adam(learningRate = 0.0001f),
            loss = MSE,
            metric = Metrics.MAE
        )

        model.fit(trainData, epochs = 6000, batchSize = 32)

        val evaluation = model.evaluate(testData)
        println("Test Loss: ${evaluation.lossValue}, Test MAE: ${evaluation.metrics[Metrics.MAE]}")

        val xTest = testData.x
        val predictions = xTest.map { model.predictSoftly(it) }
        val yTest = testData.y

        println("Predictions vs. True Values:")
        for (i in predictions.indices) {
            println("Real: ${yTest[i]}, Predicted: ${predictions[i].joinToString(",")}, From: ${xTest[i].joinToString(",")}")
        }
    }
}