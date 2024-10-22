package org.jesperancinha.oga;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import static org.nd4j.linalg.activations.Activation.RELU;
import static org.nd4j.linalg.activations.Activation.SOFTMAX;

@Disabled
public class OgaDeepLearning4JRunnerTest {

    @Test
    public void testOgaDeepLearning4JRunner() throws IOException {
        // Image dimensions and settings
        int height = 28; // Image height
        int width = 28; // Image width
        int channels = 1; // Grayscale
        int outputNum = 10; // Adjust according to your classes (e.g., 10 for Fashion-MNIST)
        int batchSize = 32; // Number of examples per batch
        int seed = 123; // Random seed for reproducibility

        // Path to the dataset
        File mainPath = new File("fashion-images"); // Update to your dataset path

        if (!mainPath.exists()) {
            throw new RuntimeException("Could not find fashion-images");
        }

        // FileSplit to load images
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, new Random());

        // Label generator: extract labels from the folder names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        // Initialize the ImageRecordReader
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(fileSplit);

        // Create a DataSetIterator
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        while (dataIter.hasNext()) {
            DataSet ds = dataIter.next();
            try {
                ds.getFeatures().reshape(batchSize, height * width * channels);
            }catch (Exception e) {
                e.printStackTrace();
            }
        }

        // Normalize pixel values to 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        // Build the model configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001)) // Optimizer
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width * channels) // Input size (flattened)
                        .nOut(128) // Number of neurons in this layer
                        .activation(RELU) // Activation function
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(128) // Number of inputs for the output layer
                        .nOut(outputNum) // Number of classes
                        .activation(SOFTMAX) // Activation function for the output layer
                        .build())
                .build();

        // Initialize the MultiLayerNetwork with the configuration
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init(); // Initialize the model

        // Train the model
        int numEpochs = 10; // Number of training epochs
        for (int i = 0; i < numEpochs; i++) {
            model.fit(dataIter);
            dataIter.reset(); // Reset the iterator for the next epoch
            System.out.println("Epoch " + (i + 1) + " completed.");
        }

        System.out.println("Model training completed.");
    }
}
