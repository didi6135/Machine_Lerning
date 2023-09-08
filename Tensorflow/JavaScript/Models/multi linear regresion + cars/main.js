
const demo = document.getElementById('demo')
// ------------------------------------------------------------------------
// -------------------- Function to extract specific data -------------------------
// ------------------------------------------------------------------------


// get only the data that we need from the json file
const extractData = (data) => {
    // console.log(`x: ${data.Horsepower} y: ${data.Miles_per_Gallon}`)

    // return the data as object with X and Y
    // that thay get the data that we need to tha model
    return {x: data.Horsepower, y: data.Miles_per_Gallon}
}

// ------------------------------------------------------------------------
// -------------------- Function to remove errors -------------------------
// ------------------------------------------------------------------------

// remove error from the data that we get 
// to make our model clean from worng data
const removeErrors = (data) => {
    return data.x !== null && data.y !== null
}

// ------------------------------------------------------------------------
// -------------------- Function to fetch data -------------------------
// ------------------------------------------------------------------------


// fetch the data from the json file 
const fetchData = async () => {
    const getData = await fetch('carsData.json')
    let values = await getData.json()

    // mapping the data with --> function extractData()
    // filter the error from the data with --> function removeErrors()
    values = values.map(extractData).filter(removeErrors)

    // move the data to the chart
    // values --> the all data
    // demo --> the div that create the chart
    drawData(values, demo)
    
    // Create model by calling the function createModel()
    const model = createModel();

    // convert data to tensor
    const tensorData = convertToTensor(values);

    // Show summery of the model in tabel
    tfvis.show.modelSummary({name: 'Model Summary'}, model);
    
    
    // Get the inputs value and the labels value from the model
    const {inputs, labels} = tensorData;

    // Train the model 
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    // in this function we test the model to predict good answer
    testModel(model, values, tensorData);
    
    
}

// ------------------------------------------------------------------------
// -------------------- Function to draw data -------------------------
// ------------------------------------------------------------------------


// function that draw the data to chart
const drawData =  (values) => {

    // spacialSurface --> create the surface of the chart including name for title
    const spacialSurface = {name: 'Horsepower vs MPG'}
    // function that create the chart
    // get 3 parameters
    tfvis.render.scatterplot(
        
        // 1. surface
        spacialSurface, 
        {values: values, series:['Original', 'Predicted']},
        {xLabel: 'Horsepower', yLabel: 'MPG', height: 300}    
    )

}



// ------------------------------------------------------------------------
// -------------------- Function to create model -------------------------
// ------------------------------------------------------------------------

// In this function we create  model 
const createModel = () => {

    const model = tf.sequential()
    model.add(tf.layers.dense({inputShape: [1], units: 1,  useBias: true}))
    model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}))
    model.add(tf.layers.dense({units: 1}))
    return model
}

// ------------------------------------------------------------------------
// -------------------- Function to convert to tensor -------------------------
// ------------------------------------------------------------------------


const convertToTensor = (data) => {
    return tf.tidy(() => {
        tf.util.shuffle(data)

        const inputs = data.map(data => data.x)
        const labels = data.map(data => data.y)

        const inputsTensor = tf.tensor2d(inputs, [inputs.length, 1])
        const labelsTensor = tf.tensor2d(labels, [labels.length, 1])
    
        const inputMin = inputsTensor.min()
        const inputMax = inputsTensor.max()
        const labelMin = labelsTensor.min()
        const labelMax = labelsTensor.max()
        
        const normalizationInputs = inputsTensor.sub(inputMin).div(inputMax.sub(inputMin))
        const normalizationLabels = labelsTensor.sub(labelMin).div(labelMax.sub(labelMin))
        
        return {
            inputs: normalizationInputs,
            labels: normalizationLabels,
            inputMin,
            inputMax,
            labelMin,
            labelMax,
        }
    })
}

// ------------------------------------------------------------------------
// -------------------- Function to train model -------------------------
// ------------------------------------------------------------------------


// Function that training the model
const trainModel = async (model, inputs, labels) => {

    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    })

    const batchSize = 32;
    const epochs = 150;
    
    return await model.fit(
        inputs, 
        labels, 
        { 
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            // we enter a name for the title
            {name: 'Training Performance'},
            // give name to series
            ['loss'],
            {height: 200, callbacks:['onEpochEnd']})
    })
}

// ------------------------------------------------------------------------
// -------------------- Function to test the model -------------------------
// ------------------------------------------------------------------------


const testModel = (model, inputData, normalizationData) => {
    
    // Get the data of min & max from normalization function
    const {inputMin, inputMax, labelMin, labelMax} = normalizationData;

    const [inputs, pred] = tf.tidy(() => {
        
    const createNormalizeInputsValues = tf.linspace(0, 1, 100)

        const predictions = model.predict(
            createNormalizeInputsValues.reshape([100, 1])
            )

        const unNormalizeInputsValues = createNormalizeInputsValues
        .mul(inputMax.sub(inputMin))
        .add(inputMin);

        const unNormalizePredictionsValues = predictions
        .mul(labelMax.sub(labelMin))
        .add(labelMin);

        return [ 
            unNormalizeInputsValues.dataSync(),
            unNormalizePredictionsValues.dataSync()
        ]
    })
    // In this constant we create array from the inputs values
    const predictedPoints = Array.from(inputs).map((value, index) => {
        return{x: value, y: pred[index]}
    })

    const originalPoints = inputData.map((data) => {
        return {x: data.x , y: data.y}
    })

    tfvis.render.scatterplot(
        {name: 'Model Predictions vs Original Data'},
        {values: [originalPoints, predictedPoints], series: ['Original', 'predicted']},
        {
          xLabel: 'Horsepower',
          yLabel: 'MPG',
          height: 300
        }
      );
}

fetchData()