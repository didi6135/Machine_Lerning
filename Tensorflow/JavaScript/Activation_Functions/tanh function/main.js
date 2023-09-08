const numberInput = document.getElementById('numberInput')
const resultText = document.getElementById('result')

// this is the equation  f(x) = (2 / (1 + e^(-2x))) - 1

const tanhResult = () => {

    // get the value from input
    const value = +numberInput.value

    // f(3) = (2 / (1 + e^(-2 * 3))) - 1
    // first calculate 
    const mulValue = -2 * value

    // we make square to mulValue with e number
    const square = Math.exp(mulValue)

    // now we add 1 to the result
    const add1 = 1 + square

    // now we divide by 2 and - 1  
    const divideBy2 = (2 / add1) - 1
    
    // see the same result
    console.log(divideBy2)

    // see the same result in the console
    const result = tf.tanh(value).print()
    
    const text = `f(${value}) = (2 / (1 + ${square.toFixed(4)})) - 1 = ${divideBy2.toFixed(4)}`
    resultText.innerHTML = text
} 

const drawPlot = () => {

    const x = tf.linspace(-10, 10, 1000)
    const valuesX = x.dataSync();
    const arrX = Array.from(valuesX);

    const y = tf.tanh(arrX)
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