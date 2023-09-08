const inputValue = document.getElementById('inputValue') 
const resultInput = document.getElementById('resultInput')


// all the number above 0 = will be the value of number
// all number below 0 = will be 0
const tensor = tf.tensor1d([-1])

tensor.relu().print()


const getValue = () => {
    const value = +inputValue.value
    const tensor = tf.tensor1d([value])

    const result = tensor.relu()

    console.log(result.dataSync()[0])

    result.print()
    resultInput.innerHTML = `result by using reLU function ${result.dataSync()}`

}

// be attention if you use reLU function remember that all negative numbers
// will be reset to 0 and the same neuron will not be updated
// and all the numbers above the 0 will get the full number 
// that will make a bug changes in weight 





