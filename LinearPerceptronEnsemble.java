import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import java.util.Arrays;
import java.util.Random;

/** Class modelling an ensemble of EnhancedLinearPerceptron objects
 *  The default size of the ensemble is 50.
 *  50% of the attributes are selected by each perceptron, and the attributes not selected are stored in a matrix.
 */
public class LinearPerceptronEnsemble {

    private int size = 50;
    private EnhancedLinearPerceptron[] ensemble;
    private double proportionAttribs = 0.5;
    private Attribute[][] attribsUsed;

    /** Generates a random index of Attribute to select and makes sure it has not already been selected
     *
     * @param numAttribs the number of attributes in the dataset
     * @param indices the already selected indices
     * @return the index of the attribute to select
     */
    private int generateRandomIndex(int numAttribs, int[] indices){
        Random rnd = new Random();
        int randomIndex = rnd.nextInt(numAttribs);
        int index = randomIndex;
        if (Arrays.stream(indices).anyMatch(j -> j == randomIndex)){
            index = generateRandomIndex(numAttribs, indices);
        }
        return index;
    }

    /** Selects which attributes to use by generating random indices, deleting the attributes at those indices,
     * and storing the deleted attributes in an array.
     *
     * @param data the data to carry out attribute selection on
     * @param proportion the proportion of attributes to select
     * @return an array of non-selected Attribute objects
     */
    public Attribute[] selectAttribs(Instances data, double proportion){
        int numAttribs = data.numAttributes() - 1;
        int numToSelect = (int)(numAttribs*(1-proportion));
        int[] randomIndices = new int[numToSelect];

        for (int i = 0; i < numToSelect; i++){
            int rand = generateRandomIndex(numAttribs, randomIndices);
            randomIndices[i] = rand;
        }

        Attribute[] selected = new Attribute[numToSelect];
        for (int i = 0; i < randomIndices.length; i++){
            int ind = randomIndices[i];
            Attribute select = data.attribute(ind);
            if (select.index() < data.numAttributes()-1) {
                selected[i] = select;
            }
        }

        return selected;
    }

    /** Builds each perceptron using attribute selection and adds them to the array of perceptron objects stored
     * by the ensemble.
     *
     * @param data
     * @throws Exception
     */
    public void buildClassifier(Instances data) throws Exception {
        this.ensemble = new EnhancedLinearPerceptron[this.size];
        this.attribsUsed = new Attribute[this.size][data.numAttributes()];

        for (int i = 0; i < this.ensemble.length; i++) {
            EnhancedLinearPerceptron perceptron = new EnhancedLinearPerceptron();
            Attribute[] selected = selectAttribs(data, this.proportionAttribs);
            attribsUsed[i] = selected;

            Instances modifiedData = data;
            for (Attribute a : attribsUsed[i]){
                if (a.index() < data.numAttributes() - 1) {
                    modifiedData.deleteAttributeAt(a.index());
                }
            }

            perceptron.buildClassifier(modifiedData);

            this.ensemble[i] = perceptron;
        }
    }

    /** Predicts the class of an instance, by classifying it with each perceptron object,
     * and using a majority vote to find the class.
     *
     * @param instance the instance to classify
     * @return the predicted class
     */
    public double classifyInstance(Instance instance){
        double predictedClass;
        int[] countVotes = new int[2];

        for (int i = 0; i < this.ensemble.length; i++) {
            EnhancedLinearPerceptron perceptron = this.ensemble[i];

            Instance modified = instance;
            for (Attribute a : attribsUsed[i]){
                if (a.index() < this.ensemble[0].data.numAttributes() - 1) {
                    modified.deleteAttributeAt(a.index());
                }
            }

            double classPredicted = perceptron.classifyInstance(modified);
            if (classPredicted == 0) {
                countVotes[0]++;
            }
            else {
                countVotes[1]++;
            }
        }

        if (countVotes[0] > countVotes[1]){
            predictedClass = 0;
        }
        else {
            predictedClass = 1;
        }

        return predictedClass;
    }

    /** Calculates the distribution of votes for each class for an instance
     *
     * @param instance the instance to classify
     * @return an array containing the proportion of votes for each class
     */
    public double[] distributionForInstance(Instance instance){
        double[] distribution = new double[2];
        int[] countVotes = new int[2];

        for (int i = 0; i < this.ensemble.length; i++) {
            EnhancedLinearPerceptron perceptron = this.ensemble[i];

            Instance modified = instance;
            for (Attribute a : attribsUsed[i]){
                if (a.index() < this.ensemble[0].data.numAttributes() - 1) {
                    modified.deleteAttributeAt(a.index());
                }
            }

            double classPredicted = perceptron.classifyInstance(modified);
            if (classPredicted == -1) {
                countVotes[0]++;
            }
            else {
                countVotes[1]++;
            }
        }

        distribution[0] = (double) countVotes[0]/this.size;
        distribution[1] = (double) countVotes[1]/this.size;

        return distribution;
    }

    public static void main(String[] args) throws Exception {

        //Testing carried out
        /*Instances trainingData = main.loadClassificationData("train_part_one.arff");

        LinearPerceptronEnsemble ensemble = new LinearPerceptronEnsemble();
        ensemble.size = 50;
        ensemble.proportionAttribs = 0.5;
        ensemble.buildClassifier(trainingData);
        trainingData.setClassIndex(trainingData.numAttributes()-1);

        Instances testData = main.loadClassificationData("test_part_one.arff");
        for (Instance data : testData){
            System.out.println("classify instance = " + ensemble.classifyInstance(data));
        }
        System.out.println();
        for (Instance data : testData){
            double[] distribution = ensemble.distributionForInstance(data);
            System.out.print("distribution for instance = ");
            for (double d : distribution){
                System.out.print(d + ", ");
            }
            System.out.println();
        }*/

    }

}
