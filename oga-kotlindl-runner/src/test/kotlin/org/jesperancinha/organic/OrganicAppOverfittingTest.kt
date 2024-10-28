package org.jesperancinha.organic

import org.jesperancinha.organic.data.generateSyntheticData
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations.Linear
import org.jetbrains.kotlinx.dl.api.core.activation.Activations.Relu
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics.MAE
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.junit.jupiter.api.Test
import kotlin.random.Random

class OrganicAppOverfittingTest {

    @Test
    fun `should run an overfitting test`() {
        val (trainData, testData) = generateSyntheticData()
        val model = Sequential.of(
            Input(1),
            Dense(512, activation = Relu),
            Dense(256, activation = Relu),
            Dense(128, activation = Relu),
            Dense(64, activation = Relu),
            Dense(32, activation = Relu),
            Dense(1, activation = Linear)
        )
        model.compile(
            optimizer = Adam(learningRate = 0.001f),
            loss = Losses.MAE,
            metric = MAE
        )
        val epochs = 5000
        val batchSize = 32

        val trainingLosses = mutableListOf<Double>()
        val trainingMAEs = mutableListOf<Double>()

        for (epoch in 1..epochs) {
            model.fit(trainData, epochs = 1, batchSize = batchSize)
            val trainEvaluation = model.evaluate(trainData)
            trainingLosses.add(trainEvaluation.lossValue)
            trainingMAEs.add(trainEvaluation.metrics[MAE]!!)
            println("Epoch $epoch: Training Loss: ${trainEvaluation.lossValue}, Training MAE: ${trainEvaluation.metrics[MAE]}")
        }
        val evaluation = model.evaluate(testData)
        println("Test Loss: ${evaluation.lossValue}, Test MAE: ${evaluation.metrics[MAE]}")

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
