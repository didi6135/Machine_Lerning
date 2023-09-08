


const run = async() => {

    const csvUrl = 'Salary_Data.csv'

    const csvData = tf.data.csv(
        csvUrl, {
            columnConfigs: {
                Salary: {
                    isLabel: true
                }
            },
            hasHeader: true,
            delimiter: ','
        })

    console.log(csvData)
    const numOfFeature = (await csvData.columnNames()).length - 1
    console.log(numOfFeature)

    const prapareData = csvData.map(({xs, ys}) => {
        // console.log(xs)

        return {
            xs: Object.values(xs),
            ys: Object.values(ys)
        }
    }).batch(10)

const predictTensor = tf.tensor([[6,148,72,35,0,33.6,0.627,50]])
predictTensor.print()

    const model = tf.sequential()
    model.add(tf.layers.dense({
        inputShape:[numOfFeature],
        units: 64
    }))
    model.add(tf.layers.dense({
        units: 32
    }))
    model.add(tf.layers.dense({
        units: 1
    }))


    model.compile({
        optimizer: tf.train.adam(0.00001),
        loss: 'meanSquaredError'
    })

await model.fitDataset(prapareData, {
    epochs: 100,
    callbacks: {
        onEpochEnd: async (epoch, logs) => {
            console.log(epoch + ':' + logs.loss)
        }
    }
})

const predict = model.predict(predictTensor)
predict.print()


}
run()