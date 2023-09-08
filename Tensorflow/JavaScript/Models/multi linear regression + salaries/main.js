
// Array to store only one kind of each type
const jobTitleLabelList = [];
const EducationLevelLabelList = [];
const GenderList = []

// array for all index 
const jobTitleIndex = []
const EducationLevelLabelIndex = []
const GenderIndex = []

const age = []
const YearsOfExperience = []
const Salary = []


const prepareData = async () => {
    
    const createNewSet = new Set();
    
    const csvUrl = 'Salary_Data.csv'
    const dataset = tf.data.csv(csvUrl);

        await dataset.forEachAsync((example) => {
        getExample(example, 'EducationLevel', EducationLevelLabelList, createNewSet)
        getExample(example, 'JobTitle', jobTitleLabelList, createNewSet)
        getExample(example, 'Gender', GenderList, createNewSet)
            

        getDataFromAgeAndYear(example, 'Age', age)
        getDataFromAgeAndYear(example, 'YearsOfExperience', YearsOfExperience)
        getDataFromAgeAndYear(example, 'Salary', Salary)
    })
    
        
        await dataset.forEachAsync((row) => {
            getIndexForAll(row, jobTitleIndex, jobTitleLabelList, 'JobTitle')
            getIndexForAll(row, GenderIndex, GenderList, 'Gender')
            getIndexForAll(row, EducationLevelLabelIndex, EducationLevelLabelList, 'EducationLevel')
        })

      
        const test = getAllData()
        return [test, Salary]
            

}

const getExample = (example, nameRow, arr, createNewSet) => {
    
    const columnValue = example[nameRow]
    
    if(!createNewSet.has(columnValue)) {
        arr.push(columnValue)
        createNewSet.add(columnValue)
    }
}

// get data from Age & year of Experience
const getDataFromAgeAndYear = (example, nameRow, arr) => {
    const columnValue = example[nameRow]
    arr.push(columnValue)
    
}


// get all the index of all category
const getIndexForAll = (example, arr, list, nameRow) => {
    const columnValue = example[nameRow]
    arr.push(list.indexOf(columnValue))
}


const normalizeData = (data) => {

    const min = tf.min(data);
    const max = tf.max(data);
    const normalizedData = data.sub(min).div(max.sub(min));
    return normalizedData;
  };



// convert all data after change string to number into one tensor
const getAllData = () => {

    const normalizedAge = normalizeData(tf.tensor2d([age]));
    const normalizedGender = normalizeData(tf.tensor2d([GenderIndex]));
    const normalizedEducationLevel = normalizeData(tf.tensor2d([EducationLevelLabelIndex]));
    const normalizedJobTitle = normalizeData(tf.tensor2d([jobTitleIndex]));
    const normalizedYearsOfExperience = normalizeData(tf.tensor2d([YearsOfExperience]));

    const tensorConcat = tf.concat([
        normalizedAge, 
        normalizedGender, 
        normalizedEducationLevel, 
        normalizedJobTitle, 
        normalizedYearsOfExperience
    ])

    return tensorConcat
}


const predictSalary = () => {
    const ageInput = document.getElementById('ageInput')
    const genderInput = document.getElementById('genderInput').selectedIndex
    const educationInput = document.getElementById('educationInput').selectedIndex
    const jobTitleInput = document.getElementById('jobTitleInput').selectedIndex
    const yearsInput = document.getElementById('yearsInput')

    
    const inputTensor = tf.tensor2d([[
        parseInt(ageInput.value),
        genderInput,
        educationInput,
        jobTitleInput,
        parseInt(yearsInput.value)
    ]])

    return inputTensor
}


const trainModel = async() => {
     
    const [tensorConcat, Salary] = await prepareData()

    const tensorSalary = tf.tensor2d(Salary, [Salary.length, 1])
    const mormolizeSalary = normalizeData(tensorSalary)


    const getInput = predictSalary()
    getInput.print()

    const predictTensor = getInput
    const minPredict = tf.min(predictTensor)
    const maxPredict = tf.max(predictTensor)

    const mormolizePredict = normalizeData(predictTensor)

    const transposedConcat = tf.transpose(tensorConcat);


    const model = tf.sequential()

    model.add(tf.layers.dense({
        inputShape: [5], 
        units: 5, 
        activation: 'sigmoid', 
        useBias: true,
        kernelInitializer: 'heUniform'
    }))

    
    // model.add(tf.layers.dense({units: 50, activation: 'relu', useBias: true}))
    // model.add(tf.layers.dense({units: 50, activation: 'relu', useBias: true}))
    
    model.add(tf.layers.dense({units: 64, activation: 'relu', useBias: true}))
    model.add(tf.layers.dense({units: 32, activation: 'relu', useBias: true}))
    model.add(tf.layers.dense({units: 1}))

    const surface = { name: 'Layer Summary'};
    tfvis.show.layer(surface, model.getLayer(undefined, 1));
    
    const learningRate = 0.0001
    const optimizer = tf.train.sgd(learningRate)
    const loss = tf.losses.meanSquaredError 

    model.compile({
        loss: loss,
        optimizer: optimizer,
    })
    
    await model.fit(transposedConcat, mormolizeSalary ,{
        epochs: 100,
        batchSize: 40,
        shuffle: true,
        callbacks: 
        tfvis.show.fitCallbacks(
            // we enter a name for the title
            {name: 'Training Performance'},
            // give name to series
            ['loss'],
            {height: 200, callbacks:['onEpochEnd']})
        // {
        //     onEpochEnd: async (epoch, logs) => {
        //         console.log(`number of epoch: ${epoch}, loss: ${logs.loss}`)
        //     }
        // }
    })

        model.summary()
        // const evalResult = model.evaluate(transposedConcat, tensorSalary);
        // const evalLoss = evalResult.dataSync()[0]; // Access the evaluation loss value
        // console.log('Evaluation loss:', evalLoss);

        const prediction = model.predict(mormolizePredict);

        const denormalizedPrediction = denormalizeData(prediction, minPredict, maxPredict);
        
        prediction.print()

        console.log(denormalizedPrediction.dataSync()[0])
        console.log(denormalizedPrediction.dataSync()[0] * 10000)
        transposedConcat.dispose()
        tensorSalary.dispose()
    
}
const denormalizeData = (normalizedData, min, max) => {
    const denormalizedData = normalizedData.mul(max.sub(min)).add(min);

    return denormalizedData;
  };







