


const result = document.getElementById('result')
const eluResult = document.getElementById('eluResult')

// Elu function work with if
// if the number grate then 0 so it's return the number
// if the number is small then 0 so it's make this f(x) = α * (e^x - 1)
const getElu = () => {
    // Define hyperparameter that determine the slope of the negative value
    const alpha = 1.0; 
    if (eluResult.value > 0) {
      const resultElu = eluResult.value;
      result.innerHTML = `Elu Result: f(${resultElu}) = ${resultElu} `
    } else {
      const resultElu = alpha * (Math.exp(eluResult.value) - 1)
      result.innerHTML = `Elu Result: ${alpha} * (e^x - 1) = ${resultElu}`

    }
}