import java.util.*
import kotlin.math.max

class Neuron {
    var weights = mutableListOf<Double>()

    // активационная функция
    fun activate(input: List<Double>): Double {
        if (weights.isEmpty()) {
            initWeightsByNull(input.size + 1)
        } else if (input.size + 1 != weights.size) { // инициализация весов в случае изменения топологии сети
            weights = MutableList(input.size + 1, { if (it < weights.size) weights[it] else 0.0 })
        }
        val x = listOf(1.0, *input.toTypedArray()) // добавляем вход 1 для смещения
        return max(0.0, sum(x))
    }

    fun setRandomWeights(size: Int) {
        weights = initWeights(size)
    }

    fun initWeightsByNull(size: Int) {
        weights = MutableList(size, { 0.0 })
    }

    // сумматор
    fun sum(input: List<Double>): Double {
        var sum = 0.0
        for (i in 0 until input.size) {
            sum += weights[i] * input[i]
        }
        return sum
    }

    // инициализация весов случайными значениями
    private fun initWeights(inputSize: Int) = MutableList(inputSize, { 0.5 - Random().nextDouble() })

    fun clone() = Neuron().also { it.weights = weights.map { it }.toMutableList() }
}

class Layer(size: Int=0) {
    val neurons = MutableList(size, { Neuron() })
    fun activate(input: List<Double>) = neurons.map { it.activate(input) }

    fun cnn(input: List<List<Double>>, matrixDivider: MatrixDivider): List<List<Double>> {
        val x = input.map {
            matrixDivider.divide(it)
        }
        return x.flatMap { l ->
            neurons.map { kernel -> l.map {
                if (kernel.weights.size != it.size) kernel.setRandomWeights(it.size)
                kernel.sum(it)
            } }
        }
    }

    companion object {
        fun relu(x: List<Double>) = x.map { max(it, 0.0) }

        fun pool(x: List<Double>, matrixDivider: MatrixDivider) =  matrixDivider.divide(x).map { it.max()!! }

        fun softmax(x: List<Double>): List<Double> {
            val y = x.map { if (it < 0) 0.0 else it }
            val sum = x.sum()
            if (sum == 0.0) return List(x.size, { 1.0/x.size })
            return y.map { it/y.sum() }
        }
    }

    fun clone() = Layer().also { it.neurons.addAll(neurons.map { it.clone() }.toMutableList()) }
}

class MatrixDivider(side: Int, size: Int, stride: Int) {
    private val positions: List<List<Int>>
    init {
        positions = (0..side-size step stride).flatMap { posY ->
            (0..side-size step stride).map { posX ->
                (0 until size).flatMap { i -> (0 until size).map { posY*side + posX + it + i * side } }
            }
        }
    }

    fun <E>divide(x: List<E>) = positions.map { it.map { x[it] } }
}