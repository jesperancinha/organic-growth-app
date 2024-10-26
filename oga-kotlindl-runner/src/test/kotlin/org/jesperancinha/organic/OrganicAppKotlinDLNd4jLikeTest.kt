package org.jesperancinha.organic

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.activation.Activations.Relu
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.junit.jupiter.api.Test

class OrganicAppKotlinDLNd4jLikeTest {

    @Test
    fun `should test Nd4j`() {
        val inputData = arrayOf(
            floatArrayOf(0.0f, 0.0f),
            floatArrayOf(0.0f, 1.0f),
            floatArrayOf(1.0f, 0.0f),
            floatArrayOf(1.0f, 1.0f)
        )

        val labels = floatArrayOf(
            0.0f,
            1.0f,
            1.0f,
            0.0f
        )

        val dataset = OnHeapDataset.create(
            featuresGenerator = { inputData },
            labelGenerator = { labels }
        )

        val model = Sequential.of(
            Input(2),
            Dense(2, activation = Relu),
            Dense(1, activation = Activations.Sigmoid)
        )

        model.compile(loss = Losses.MSE, optimizer = Adam(), metric = Metrics.ACCURACY)

        for (i in inputData.indices) {
            model.fit(dataset, epochs = 100) // Or set a different batch size or epochs
        }

        val predictions = inputData.map { model.predict(floatArrayOf(it[0], it[1])) }

        println("Predictions:")
        predictions.forEach { println(it.toString()) }
    }

}