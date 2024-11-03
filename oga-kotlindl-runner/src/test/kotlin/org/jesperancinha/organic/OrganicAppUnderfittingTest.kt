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

class OrganicAppUnderfittingTest {

    @Test
    fun `should run a model created with underfitting`() {
        val (trainData, testData) = generateSyntheticData()
        val model = Sequential.of(
            Input(1),
            Dense(64, activation = Relu),
            Dense(1, activation = Linear)
        )
        model.compile(
            optimizer = Adam(),
            loss = MSE,
            metric = Metrics.MAE
        )
        model.fit(trainData, epochs = 100, batchSize = 32)
        val evaluation = model.evaluate(testData)
        println("Test MSE Loss: ${evaluation.lossValue}, Test MAE metric: ${evaluation.metrics[Metrics.MAE]}")

        val xTest = testData.x
        val predictions = xTest.map { model.predictSoftly(it) }
        val yTest = testData.y

        println("Predictions vs. True Values:")
        for (i in predictions.indices) {
            println("Real: ${yTest[i]}, Predicted: ${predictions[i][0]}")
        }
    }
}