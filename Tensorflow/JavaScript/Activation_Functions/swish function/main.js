const numberInput = document.getElementById('numberInput')
const resultText = document.getElementById('result')

// this is the equation  f(x) = x / (1 + e^(-x))

const swishResult = () => {

    // get the value from input
    const value = +numberInput.value

    // short equation
    const test = value / (1 + Math.exp(-value))

    const exponent = Math.exp(-value)
    const add = 1 + exponent
    const sum = value / add

    console.log(sum)

    // see the same result
    const text = `f(${value}) = ${value} / (1 + e^(-${value})) = ${sum}`
    resultText.innerHTML = text
} 


const getSwishResult = (value) => {

    let negative = []

    value.map(num => negative.push(num * -1))
    console.log(negative)
    
    const exponent = tf.exp(negative)
    const test = exponent.dataSync()
    console.log(test)

    const add = tf.add(exponent, 1)
    
    const sum = tf.div(value, add)

    return sum
} 


const drawPlot = () => {

    const x = tf.linspace(-10, 10, 1000)
    const valuesX = x.dataSync();
    const arrX = Array.from(valuesX);

    const y = getSwishResult(arrX)
    const valuesY = y.dataSync();
    console.log(valuesY)
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