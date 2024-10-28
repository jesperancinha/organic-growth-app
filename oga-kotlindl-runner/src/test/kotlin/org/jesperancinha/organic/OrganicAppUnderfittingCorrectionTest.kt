package org.jesperancinha.organic

import org.jesperancinha.organic.data.generateSyntheticData
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.junit.jupiter.api.Test
import kotlin.random.Random

class OrganicAppUnderfittingCorrectionTest {

    @Test
    fun `should run a model created with corrected underfitting`() {
        val (trainData, testData) = generateSyntheticData()

        val model = Sequential.of(
            Input(1),
            Dense(128, activation = Activations.Relu),
            Dense(64, activation = Activations.Relu),
            Dense(32, activation = Activations.Relu),
            Dense(16, activation = Activations.Relu),
            Dense(1, activation = Activations.Linear)
        )

        model.compile(
            optimizer = Adam(learningRate = 0.01f),
            loss = Losses.MSE,
            metric = Metrics.MAE
        )

        model.fit(trainData, epochs = 2000, batchSize = 32)

        val evaluation = model.evaluate(testData)
        println("Test Loss: ${evaluation.lossValue}, Test MAE: ${evaluation.metrics[Metrics.MAE]}")

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