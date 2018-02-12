import java.util.*
import kotlin.math.max

class Neuron {
    var weights = mutableListOf<Double>()

    // активационная функция
    fun activate(input: List<Double>): Double {
        if (input.size + 1 != weights.size) { // инициализация весов в случае изменения топологии сети
            setRandomWeights(input.size + 1)
        }
        val x = listOf(1.0, *input.toTypedArray()) // добавляем вход 1 для смещения
        return max(0.0, sum(x))
    }

    fun setRandomWeights(size: Int) {
        weights = initWeights(size)
    }

    // сумматор
    fun sum(input: List<Double>) = weights.mapIndexed { i, w -> input[i] * w }.sum()

    // инициализация весов случайными значениями
    private fun initWeights(inputSize: Int) = MutableList(inputSize, { 0.5 - Random().nextDouble() })
}

class Layer(size: Int=0) {
    val neurons = MutableList(size, { Neuron() })
    fun activate(input: List<Double>) = neurons.map { it.activate(input) }

    fun cnn(input: List<List<Double>>, filterSize: Int, stride: Int): List<List<Double>> {
        val x = input.map {
            Encoder.divideToMatrix(it, filterSize, stride)
        }
        val y = (0 until x.first().size).map { i->
            (0 until x.size).flatMap { j ->
                x[j][i]
            }
        }
        return neurons.map { kernel -> y.map {
            if (kernel.weights.size != it.size) kernel.setRandomWeights(it.size)
            kernel.sum(it)
        } }
    }

    companion object {
        fun relu(x: List<Double>) = x.map { max(it, 0.0) }

        fun pool(x: List<Double>) =  Encoder.divideToMatrix(x, 2, 2).map { it.max()!! }

        fun softmax(x: List<Double>): List<Double> {
            val sum = x.sum()
            if (sum == 0.0) return List(x.size, { 1.0/x.size })
            return x.map { it/x.sum() }
        }
    }

}

object Encoder {
    private fun <T> conv(input: List<T>, x0: Int, side: Int, matrixSize: Int) = List(side, { i: Int -> List(side, { input[x0 + it + matrixSize * i] }) }).flatMap { it }

    fun <T>divideToMatrix(x: List<T>, size: Int, shift: Int): List<List<T>> {
        val inputSide = Math.sqrt(x.size.toDouble()).toInt()
        return (0..inputSide-size step shift).map {row ->
            (0..inputSide - size step shift)
                    .map { conv(x, it +row*inputSide, size, inputSide) }
        }.flatMap { it }
    }
}