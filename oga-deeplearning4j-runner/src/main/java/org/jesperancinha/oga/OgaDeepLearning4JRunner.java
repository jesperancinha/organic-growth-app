package org.jesperancinha.oga;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class OgaDeepLearning4JRunner {
    public static void main(String[] args) throws Exception {
        int batchSize = 32;
        int outputNum = 10;
        int numEpochs = 10;
        int seed = 123;

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(28 * 28)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(1000)
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(500)
                        .nOut(200)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(200)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        model.setListeners(new ScoreIterationListener(100));

        System.out.println("Training the model...");
        for (int i = 0; i < numEpochs; i++) {
            model.fit(mnistTrain);
            System.out.println("Epoch " + (i + 1) + " complete.");
        }

        File modelFile = new File("mnist-model.zip");
        model.save(modelFile, true);
        System.out.println("Model saved to " + modelFile.getAbsolutePath());

        System.out.println("Loading and classifying a single image...");
        File inputFile = new ClassPathResource("/one.png").getFile();

        NativeImageLoader loader = new NativeImageLoader(28, 28, 1);
        INDArray image = loader.asMatrix(inputFile);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);

        INDArray output = model.output(image.reshape(1, 28 * 28));
        int predictedDigit = Nd4j.argMax(output, 1).getInt(0);

        System.out.println("Predicted digit: " + predictedDigit);
    }
}
