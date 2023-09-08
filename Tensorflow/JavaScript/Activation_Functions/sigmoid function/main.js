const inputValue = document.getElementById('inputValue') 
const resultInput = document.getElementById('resultInput')


// all the number above 0 = to close to 1 
// all number below 0 = to close to 0
const tensor = tf.tensor1d([-1])

tensor.sigmoid().print()


const getValue = () => {
    let text =``
    const value = +inputValue.value
    const tensor = tf.tensor1d([value])

    const result = tensor.sigmoid()

    console.log(result.dataSync()[0])

    if( result.dataSync()[0] === 1 ) {
      text +=  `number: ${value} you have problme of vanish gradient`
    } else if(result.dataSync()[0] === 0) {
        text +=  `number: ${value} you have problme of vanish gradient`
    }
    result.print()
    resultInput.innerHTML = `result by using sigmoid function ${result.dataSync()} <br> ${text}`
}

// be attention if you try to check number that big 17 you will get result 1
// that becuse the sigmoid he have a problme of vanish gradient


const model = tf.sequential();

model.add(tf.layer)