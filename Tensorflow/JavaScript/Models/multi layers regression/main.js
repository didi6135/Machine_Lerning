const orginalData = document.getElementById('orginalData')
const predictedData = document.getElementById('predictedData')
const lossFunction = document.getElementById('lossFunction')
const weight = document.getElementById('weight')


const epocheNumber = document.getElementById('epocheNumber')
const learningRate = document.getElementById('learningRate')

const createModel = () => {
        // Here we build tensor with 5 feature inside each tensor
    const inputData = tf.tensor2d([
        [5, 10, 15, 20, 25], 
        [40, 45, 50, 55, 60], 
        [100, 105, 110, 115, 120], 
        [260, 265, 270, 275, 280]]);

        console.log(inputData.shape)
    // Here we build the label that we want to get in each tensor
    // in this example we want to get in each tensor the next number
    const targetData = tf.tensor2d([
        [30, 35], 
        [65, 70], 
        [125, 130], 
        [285, 290],
    ]);

    // Define the model architecture
    const model = tf.sequential();
    // i the first layer we define all necessary things
    // we define 5 units (neurons) we can make it 1 too
    // we define the shape of the data that come in that is 5 each
    // we can active the bias to see 
    model.add(tf.layers.dense({ units: 5, inputShape: [5], useBias: true}));

    // we can play with the numbers of neurons and see a lot of result
    model.add(tf.layers.dense({ units: 10}));

    model.add(tf.layers.dense({ units: 2}));


    // function to get the weights of all neuron
    const getWeightFromLayers = (model) => {

        model.layers.forEach((layer) => {
           const layerWeight = layer.getWeights()
        
           layerWeight.forEach((weights) => {
               const test = weights.arraySync()
               console.log(test)
           })
        })
    }

    // Configure the model
        let LR
        if(learningRate.value) {
            LR = +learningRate.value
        } else {
            LR = 0.01
        }

        const opt = tf.train.adam(LR)
        const error = tf.losses.meanSquaredError

        model.compile({ 
            loss: error, 
            optimizer: opt 
        });

        return [inputData, targetData, model]
}



const trainModel = async () => {

    const [inputData, targetData, model] = createModel()
    const tensor = tf.tensor2d([[900, 905, 910, 915, 920]])
    
    const convert = Array.from(tensor.arraySync())

    orginalData.innerHTML = `The original data: [${convert}]`
    predictedData.innerHTML = 'training...'

            let epocheNum
            if(epocheNumber.value) {
                epocheNum = +epocheNumber.value
            } else {
                epocheNum = 1000
            }

        // Train the model
        await model.fit(inputData, targetData, { 
            epochs: epocheNum,
            // batchSize: 10,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    // getWeightFromLayers(model);
                    lossFunction.innerHTML = `Number of epoch: ${epoch} Sum loss:  ${logs.loss}`;
                },
              }
        })
            
        // Use the trained model to make predictions
        const prediction = model.predict(tensor);
            
        const convertPredict = Array.from(prediction.arraySync()[0]) 

        predictedData.innerHTML = `${convertPredict[0].toFixed(2)} | ${convertPredict[1].toFixed(2)} `
        prediction.print();
     
}




