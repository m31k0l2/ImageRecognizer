import java.util.*
import kotlin.math.sqrt

interface Network {
    fun activate(x: Image): List<Double>
    fun clone(): Network
    val layers: MutableList<Layer>
    fun dropout(layersNumbers: List<Int>, dropoutRate: Double, isRandom: Boolean=false): Network
    companion object {
        var useSigma = true
    }
}

class CNetwork: Network {
    override val layers = mutableListOf<Layer>()
    private val calcImages = mutableMapOf<Image, List<Double>>()

    companion object {
        val cnnDividers = listOf(
                MatrixDivider(28,5,1),
                MatrixDivider(12,5,1),
                MatrixDivider(4,2,1),
                MatrixDivider(2,2,1)
        )

        val poolerDividers = listOf(
                MatrixDivider(24,2,2),
                MatrixDivider(8,2,2),
                null,
                null
        )
    }

    fun activateConvLayers(x: Image): List<Double> {
        var y = x.colorsMatrix
        for (i in 0..3) {
            val layer = layers[i] as CNNLayer
            y = layer.activate(y)
        }
        return norm(y.flatten())
    }

    private fun activateFullConnectedLayers(x: List<Double>): List<Double> {
        var o = x
        for (i in 4..5) {
            val layer = layers[i] as FullConnectedLayer
            o = layer.activate(o)
        }
        return o
    }

    override fun activate(x: Image): List<Double> {
        calcImages[x]?.let { return it }
        var o = x.o ?: activateConvLayers(x)
        o = activateFullConnectedLayers(o)
        val r = softmax(o)
        calcImages[x] = r
        return r
    }

    private fun softmax(x: List<Double>): List<Double> {
        val y = x.map { if (it < 0) 0.0 else it }
        val sum = x.sum()
        if (sum == 0.0) return List(x.size, { 1.0/x.size })
        return y.map { it/y.sum() }
    }

    private fun norm(x: List<Double>): List<Double> {
        val l = sqrt(x.map { it*it }.sum())
        if (l == 0.0) return x
        return x.map { it / l }
    }

    override fun clone() = CNetwork().also { it.layers.addAll(layers.map { it.clone() }) }

    override fun dropout(layersNumbers: List<Int>, dropoutRate: Double, isRandom: Boolean): CNetwork {
        for (l in layersNumbers) {
            for (neuron in layers[l].neurons) {
                for (i in 1..neuron.weights.size) {
                    if (Random().nextDouble() <= dropoutRate)
                        neuron.weights[Random().nextInt(neuron.weights.size)] = if (!isRandom) 0.0 else 1 - 2*Random().nextDouble()
                }
            }
        }
        return this
    }
}