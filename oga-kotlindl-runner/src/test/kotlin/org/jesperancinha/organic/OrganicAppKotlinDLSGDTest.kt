package org.jesperancinha.organic

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.activation.Activations.Relu
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics.ACCURACY
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import org.junit.jupiter.api.Test

class OrganicAppKotlinDLSGDTest {

    @Test
    fun `should run example with SGD`(){
            val (train, test) = mnist()

            // Define a simple neural network model
        val model = Sequential.of(
            Input(28, 28, 1), // Input layer for MNIST images
            Conv2D(filters = 32, kernelSize = 3, activation = Relu), // Convolutional layer
            MaxPool2D(poolSize = 2),
            Conv2D(filters = 64, kernelSize = 3, activation = Relu), // Second convolutional layer
            MaxPool2D(poolSize = 2),
            Flatten(),
            Dense(10, Activations.Softmax)

        )
                    model.use {
                // Configure the model to use the SGD optimizer
                it.compile(
                    optimizer = SGD(learningRate = 0.01f),
                    loss = org.jetbrains.kotlinx.dl.api.core.loss.Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                    metric = ACCURACY
                )

                // Train the model for 10 epochs with a batch size of 32
                it.fit(dataset = train, epochs = 10, batchSize = 32)

                // Evaluate the model on the test set
                val evaluationResult = it.evaluate(dataset = test, batchSize = 32)
                println("Test accuracy: ${evaluationResult.metrics[ACCURACY]}")
            }
        }
}