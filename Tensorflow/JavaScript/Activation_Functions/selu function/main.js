const numberInput = document.getElementById('numberInput')
const resultText = document.getElementById('result')

// this is the equation  f(x) = λ * (max(0, x) + alpha * (exp(min(0, x)) - 1))

// x < 0 ? scale * alpha * (exp(x) - 1) : scale * x

// 3 < 0 ? scale * alpha * (exp(3) - 1) : scale * 3


const tanhResult = () => {

    const scale = 1.05070098
    const alpha = 1.67326324

    // get the value from input
    const value = +numberInput.value

    const min = Math.min(0, value)
    const max = Math.max(0, value)
    // scale * (Math.max(0, value) + alpha * (Math.exp(Math.min(0, value)) - 1))

    // f(3) = 1.0507 * (max(0, 3) + 1.67326 * (exp(min(0, 3)) - 1))
    const firstSum = alpha * (Math.exp(min) - (1))
    const secondSum = max + firstSum
    const lastSum = scale * (secondSum)
    // console.log(lastSum)

    // see the same result in the console
    const result = tf.selu(value).print()

    const sum1 = Math.exp(min) - 1
    const sum2 = alpha * sum1
    const sum3 = max + sum2
    const sum4 = scale * sum3

    console.log(sum4)
    
    const text = `f(x) = ${scale} * (max(${max}) + ${alpha} * (exp(min(${min})) - 1)) = ${lastSum}`
    resultText.innerHTML = text
} 




const drawPlot = () => {

    const x = tf.linspace(-10, 10, 1000)
    const valuesX = x.dataSync();
    const arrX = Array.from(valuesX);

    const y = tf.selu(arrX)
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