const inputValue = document.getElementById('inputValue') 
const resultInput = document.getElementById('resultInput')


// all the number above 0 = get the same value until that is get number 6 that the top value
// all number below 0 = come to be 0
const tensor = tf.tensor1d([-1])

tensor.relu6().print()


const getValue = () => {
    const value = +inputValue.value
    const tensor = tf.tensor1d([value])

    const result = tensor.relu6()

    console.log(result.dataSync()[0])

    result.print()
    resultInput.innerHTML = `result by using reLU6 function ${result.dataSync()}`

}

// be attention if you use reLU function remember that all nagtive numbers
// will be reset to 0 and the same neuron will not be updateed
// and all the numbers above the 0 will get the full number 
// the top number that cen get is value is number 6 more then 6 will get result of 6





