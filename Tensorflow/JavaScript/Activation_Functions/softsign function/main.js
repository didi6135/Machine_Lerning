const inputValue = document.getElementById('inputValue')
const resultInput = document.getElementById('resultInput')

const getValue = () => {
    // Get value from user
    const value = +inputValue.value

    // Get the absolute of the input
    const absoluteValue = tf.abs(value)

    // Add 1 to the absolute number
    const sumAbsolutePlus1 = tf.add(absoluteValue, 1)

    // Div the value wuth sumAbsolutePlus1
    const result = tf.div(value, sumAbsolutePlus1)

    resultInput.innerHTML = `(${value}) / (1 + abs(${absoluteValue.dataSync()})) = ${result.dataSync()}`

    result.print()
}




const printValue = (num) => {

    // Get thhe absolute of the input
    const absoluteValue = tf.abs(num)

    // Add 1 to the absolute number
    const sumAbsolutePlus1 = tf.add(absoluteValue, 1)

    // Div the value wuth sumAbsolutePlus1
    const result = tf.div(num, sumAbsolutePlus1)
    return result
}


const drawPlot = () => {

    const x = tf.linspace(-10, 10, 1000)
    const valuesX = x.dataSync();
    const arrX = Array.from(valuesX);

    const y = printValue(arrX)
    const valuesY = y.dataSync();
    const arrY = Array.from(valuesY);

    const trace1 = {
        x: arrX,
        y: arrY,
        type: 'scatter'
      };

      const data = [trace1];
      
      Plotly.newPlot('myDiv', data);
    
}

drawPlot()