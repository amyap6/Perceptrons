import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import java.util.Random;

/** Class modelling a Linear Perceptron with parameter tuning improvements.
 *  The class is an extended version of LinearPerceptron.java, with added capabilities of standardising the data,
 *  choosing an off-line learning algorithm, and selecting whether to use on-line or off-line based on the
 *  cross validation error.
 *  By default, the attributes are standardised, and on-line is used without model selection
 */
public class EnhancedLinearPerceptron extends AbstractClassifier {

    private static double learningRate;
    private static int MAX_ITERATIONS = 1000;
    private double[] linearModel;

    private boolean biasTerm = false;
    public Instances data;

    private boolean STANDARDISE_ATTRIBUTES = true;
    private double[] means;
    private double[] standardDeviations;

    private boolean USE_ALTERNATIVE_ALGORITHM = false;
    private boolean MODEL_SELECTION = false;

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

    /** Standardises the data's attributes by calculating the means and standard deviations,
     * and updating the attribute values in the dataset to be zero mean and unit standard deviation
     *
     * @param data the data to be standardised
     * @return the standardised data
     */
    private Instances standardiseData(Instances data){
        Instances standardisedData = data;

        this.means = new double[data.numAttributes()];
        this.standardDeviations = new double[data.numAttributes()];

        for (int i = 0; i < data.numAttributes() - 1; i++){

            double sum = 0.0;
            for (int j = 0; j < data.numInstances(); j++){
                sum += data.get(j).value(i);
            }

            double mean = sum / (double) data.numInstances();
            this.means[i] = mean;
            double standardDeviation = 0.0;

            for (int j = 0; j < data.numInstances(); j++){
                standardDeviation += Math.pow(data.get(j).value(i) - mean, 2);
            }
            this.standardDeviations[i] = Math.sqrt(standardDeviation/data.numInstances());

            for (int j = 0; j < data.numInstances(); j++){
                double standardised = data.get(j).value(i) - means[i] / standardDeviations[i];
                standardisedData.get(j).setValue(i, standardised);
            }
        }

        return standardisedData;
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

    /** Builds the classifier on a set of training data to form a linear model
     * The weight vector is initialised to random small values, with the possible inclusion of a bias term
     * The training method then iterates over the dataset, using the off-line learning rule to update the weight vector
     * The iteration stops when the maximum number of iterations has been met, or there are no more errors.
     *
     * @param data the dataset to build a linear model on
     * @return the linear model/final weight vector
     */
    public double[] gradientDescentTraining(Instances data){
        double[] weightVector = new double[data.numAttributes() + 1];

        for (int i = 0; i < data.numAttributes(); i++){
            double w = new Random().nextDouble();
            weightVector[i] = w;
        }

        if (this.biasTerm){
            double bias = new Random().nextDouble();
            weightVector[weightVector.length - 1] = bias;
        }

        double localError;
        int iteration = 0;

        do{
            iteration++;
            double[] weightChange = new double[weightVector.length];

            double sum = 0.0;

            for (Instance i : data){

                for (int attr = 0; attr < i.numAttributes(); attr++) {
                    sum += weightVector[attr]*i.value(attr);
                }

                double predictedClass = java.lang.Math.signum(sum);
                localError = i.classValue() - predictedClass;

                for (int j = 0; j < weightVector.length - 1; j++){
                    weightChange[j] += 0.5*learningRate*localError*i.value(j);
                }
            }

            for (int j = 0; j < weightVector.length; j++){
                weightVector[j] += weightChange[j];
            }

        }
        while(iteration <= MAX_ITERATIONS);

        return weightVector;
    }

    /** Decides whether the on-line or off-line learning algorithm should be used, by building the classifier on
     * the data using each of them, calculating the cross validation error, and choosing the algorithm with the
     * lowest error
     *
     * @param train the training data being used
     * @return true if on-line should be used, false if off-line should be used
     * @throws Exception catches unhandles Weka Exceptions
     */
    private boolean selectModel(Instances train) throws Exception {
        this.linearModel = perceptronTraining(train);
        int numFolds;

        Evaluation eval = new Evaluation(train);
        if (train.numInstances() >= 10){
            numFolds = 10;
        }
        else {
            numFolds = train.numInstances();
        }
        eval.crossValidateModel(this, train, numFolds, new Random(1));
        double onlineAccuracy = eval.pctCorrect();

        this.linearModel = gradientDescentTraining(train);
        eval.crossValidateModel(this, train, numFolds, new Random(1));

        double offlineAccuracy = eval.pctCorrect();

        return onlineAccuracy > offlineAccuracy;
    }

    /** Sets the data variable if the data is continuous, then standardises the attributes if the flag is set
     * Uses model selection if the flag is set, or the off-line algorithm if that flag is set
     *
     * @param data the training data to build the classifier on
     * @throws Exception catches unhandled Weka Exceptions
     */
    public void buildClassifier(Instances data) throws Exception {
        this.getCapabilities().testWithFail(data);
        if (STANDARDISE_ATTRIBUTES){
            this.data = standardiseData(data);
        }
        else{
            this.data = data;
        }
        if (MODEL_SELECTION){
            boolean model = selectModel(this.data);
            if (model){
                this.linearModel = perceptronTraining(this.data);
            }
            else {
                this.linearModel = gradientDescentTraining(this.data);
            }
        }
        if (!USE_ALTERNATIVE_ALGORITHM) {
            this.linearModel = perceptronTraining(this.data);
        }
        else {
            this.linearModel = gradientDescentTraining(this.data);
        }
    }

    /** Predicts the class of an instance, first by applying the weight vector to the instance and standardising
     * the attributes. The class is predicted based on whether it passes the threshold value (the sum of the weight vector)
     *
     * @param instance the instance to classify
     * @return the predicted class
     */
    public double classifyInstance(Instance instance){
        double predictedClass;
        double weightedSum = 0.0;

        if (STANDARDISE_ATTRIBUTES) {
            for (int i = 0; i < instance.numAttributes() - 1; i++) {
                if (i < means.length - 1 && i < standardDeviations.length - 1) {
                    double standardised = instance.value(i) - means[i] / standardDeviations[i];
                    instance.setValue(i, standardised);
                }
            }
        }

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

        EnhancedLinearPerceptron perceptron = new EnhancedLinearPerceptron();
        perceptron.STANDARDISE_ATTRIBUTES = true;
        perceptron.MODEL_SELECTION = false;
        perceptron.USE_ALTERNATIVE_ALGORITHM = false;
        trainingData.setClassIndex(trainingData.numAttributes()-1);
        perceptron.buildClassifier(trainingData);

        for (int i = 0; i < perceptron.linearModel.length; i++){
            System.out.println("linear model=" + perceptron.linearModel[i]);
        }

        Instances testData = main.loadClassificationData("test_part_one.arff");
        for (Instance data : testData){
            System.out.println(perceptron.classifyInstance(data));
        }*/

    }

}
