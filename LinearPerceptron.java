import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import java.util.Random;

/** Class modelling a Linear Perceptron classifier
 *  Capable of building a classifier object by finding the weight vector from an on-line learning algorithm,
 *  and classifying instances using the generated linear model.
 *  A bias term may be included, and the learning rate and maximum number of iterations is modifiable
 *  The classifier should only be used with continuous data
 */
public class LinearPerceptron extends AbstractClassifier {

    private double learningRate = 1;
    private int MAX_ITERATIONS = 1000;
    private double[] linearModel;
    private boolean biasTerm = false;
    public Instances data;

    /** Uses the Weka capabilities feature to disable nominal attributes,
     * enforcing that the classifier should only be used with continuous data
     * @return the capabilities of the classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities caps = super.getCapabilities();
        caps.disable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        return caps;
    }

    /** Builds the classifier on a set of training data to form a linear model
     * The weight vector is initialised to random small values, with the possible inclusion of a bias term
     * The training method then iterates over the dataset, using the on-line learning rule to update the weight vector
     * The iteration stops when the maximum number of iterations has been met, or there are no more errors.
     *
     * @param data the dataset to build a linear model on
     * @return the linear model/final weight vector
     */
    private double[] perceptronTraining(Instances data){

        double[] weightVector = new double[data.numAttributes() + 1];

        for (int i = 0; i < data.numAttributes(); i++){
            double w = new Random().nextDouble();
            weightVector[i] = w;
        }

        if (this.biasTerm){
            double bias = new Random().nextDouble();
            weightVector[weightVector.length - 1] = bias;
        }

        boolean changes = true;
        int lastModified = -1;
        boolean currentModified;
        int iteration = 0;
        this.learningRate = 1.0;

        do {
            iteration++;
            double localError;

            for (int i = 0; i < data.numInstances(); i++){

                currentModified = false;
                Instance instance = data.instance(i);
                double sum = 0.0;

                for (int attr = 0; attr < instance.numAttributes(); attr++) {
                    sum += weightVector[attr]*instance.value(attr);
                }

                double predictedClass = java.lang.Math.signum(sum);
                localError = instance.classValue() - predictedClass;

                for (int j = 0; j < weightVector.length - 1; j++) {
                    double weightChange = 0.5*this.learningRate*localError*instance.value(j);
                    if (weightChange != 0.0 && weightChange != -0.0){
                        lastModified = i;
                        currentModified = true;
                    }
                    weightVector[j] += weightChange;
                }

                if (!currentModified && lastModified == i) {
                    changes = false;
                }
            }
        }

        while(iteration < this.MAX_ITERATIONS && changes);

        return weightVector;
    }

    /** Sets the data variable if the data is continuous
     *  Trains the perceptron to find the linear model
     *
     * @param data the training data to build the classifier on
     * @throws Exception catches unhandled Weka Exceptions
     */
    @Override
    public void buildClassifier(Instances data) throws Exception{
        this.getCapabilities().testWithFail(data);
        this.data = data;
        this.linearModel = perceptronTraining(this.data);
    }

    /** Predicts the class of an instance, first by applying the weight vector to the instance
     *  The class is predicted based on whether it passes the threshold value (the sum of the weight vector)
     *
     * @param instance the instance to classify
     * @return the predicted class
     */
    public double classifyInstance(Instance instance){
        double predictedClass;
        double weightedSum = 0.0;

        for (int i = 0; i < this.linearModel.length - 1; i++){
            instance.setValue(i, instance.value(i)*this.linearModel[i]);
            weightedSum += instance.value(i);
        }

        double threshold = 0.0;
        for (int i = 0; i < this.linearModel.length - 1; i++){
            threshold += linearModel[i];
        }

        if (weightedSum > threshold){
            predictedClass = 1;
        }
        else {
            predictedClass = 0;
        }
        return predictedClass;
    }

    public static void main(String[] args) throws Exception {

        //Testing carried out
        /*Instances trainingData = main.loadClassificationData("train_part_one.arff");

        LinearPerceptron perceptron = new LinearPerceptron();
        perceptron.biasTerm = false;
        perceptron.buildClassifier(trainingData);
        trainingData.setClassIndex(trainingData.numAttributes()-1);

        for (int i = 0; i < perceptron.linearModel.length; i++){
            System.out.println("linear model=" + perceptron.linearModel[i]);
        }

        Instances testData = main.loadClassificationData("test_part_one.arff");
        for (Instance data : testData){
            System.out.println(perceptron.classifyInstance(data));
        }*/

    }

}
