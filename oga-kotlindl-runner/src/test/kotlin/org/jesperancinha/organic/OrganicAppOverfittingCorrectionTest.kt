package org.jesperancinha.organic

import org.jetbrains.kotlinx.dl.api.core.Sequential
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
        val (trainData, testData) = generateNormalizedSyntheticData()
        val model = Sequential.of(
            Input(1),
            Dense(64, activation = Relu),
            Dropout(0.5f),
            Dense(32, activation = Relu),
            Dropout(0.5f),
            Dense(1, activation = Linear)
        )

        // Compile the model
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
    }
    fun generateNormalizedSyntheticData(): Pair<OnHeapDataset, OnHeapDataset> {
        val random = Random(42) // Seed for reproducibility
        val xTrainRaw = (0..100).map { it.toFloat() }.toFloatArray()
        val yTrainRaw = (0..100).map { (2 * it + 1 + random.nextDouble() * 10).toFloat() }.toFloatArray()
        val xTestRaw = (101..200).map { it.toFloat() }.toFloatArray()
        val yTestRaw = (101..200).map { (2 * it + 1 + random.nextDouble() * 10).toFloat() }.toFloatArray()
        val xTrain = xTrainRaw.map { floatArrayOf(it / 200f) }.toTypedArray()
        val yTrain = yTrainRaw.map { it / 200f }.toFloatArray()
        val xTest = xTestRaw.map { floatArrayOf(it / 200f) }.toTypedArray()
        val yTest = yTestRaw.map { it / 200f }.toFloatArray()
        val trainData = OnHeapDataset.create(xTrain, yTrain)
        val testData = OnHeapDataset.create(xTest, yTest)
        return Pair(trainData, testData)
    }

}
