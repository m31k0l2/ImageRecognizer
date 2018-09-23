import java.util.*
import kotlin.math.sqrt

interface Network {
    fun activate(x: Image, alpha: Double): List<Double>
    fun activateConvLayers(layers: List<CNNLayer>, x: Image): List<Double>
    fun clone(): Network
    val layers: MutableList<Layer>
    fun dropout(layersNumbers: List<Int>, dropoutRate: Double, isRandom: Boolean=false): Network
}

class CNetwork: Network {
    override val layers = mutableListOf<Layer>()
    private val calcImages = mutableMapOf<Image, List<Double>>()

    private val netClasses by lazy {
        layers.asSequence().map { it.classNet }.toSet()
    }

    companion object {
        val cnnDividers = listOf(
                MatrixDivider(28,5,1),
                MatrixDivider(12,5,1),
                MatrixDivider(4,2,1),
                MatrixDivider(2,2,1)
        )

        val poolDividers = listOf(
                MatrixDivider(24,2,2),
                MatrixDivider(8,2,2),
                null,
                null
        )
    }

    override fun activateConvLayers(layers: List<CNNLayer>, x: Image): List<Double> {
        var y = x.colorsMatrix
        layers.forEach {
            y = it.activate(y)
        }
        return norm(y.flatten())
    }

    private fun activateFullConnectedLayers(layers: List<FullConnectedLayer>, x: List<Double>, alpha: Double): List<Double> {
        var o = x
        layers.forEach {
            o = it.activate(o, alpha)
        }
        return o
    }

    private fun activateClass(classNet: String, x: Image, alpha: Double): List<Double> {
        val layers = layers.filter { it.classNet == classNet }
        val o = activateConvLayers(layers.asSequence().filter { it is CNNLayer }.map { it as CNNLayer }.toList(), x)
        return activateFullConnectedLayers(layers.asSequence().filter { it is FullConnectedLayer }.map { it as FullConnectedLayer }.toList(), o, alpha)
    }

    fun activateLayers(x: Image, alpha: Double) = netClasses.filter { it != "final" }.flatMap {
        activateClass(it, x, alpha)
    }

    override fun activate(x: Image, alpha: Double): List<Double> {
        calcImages[x]?.let { return it }
        if (netClasses.size > 1) {
            val y = x.y ?: activateLayers(x, alpha)
            val o = activateFullConnectedLayers(layers.asSequence().filter { it.classNet == "final" }.map { it as FullConnectedLayer }.toList(), y, alpha)
            val r = softmax(o)
            calcImages[x] = r
            return r
        }
        var o = x.o ?: activateConvLayers(layers.asSequence().filter { it is CNNLayer }.map { it as CNNLayer }.toList(), x)
        o = activateFullConnectedLayers(layers.asSequence().filter { it is FullConnectedLayer }.map { it as FullConnectedLayer }.toList(), o, alpha)
//        val r = softmax(o)
        calcImages[x] = o
        return o
    }

    private fun softmax(x: List<Double>): List<Double> {
        val y = x.map { if (it < 0) 0.0 else it }
        val sum = x.sum()
        if (sum == 0.0) return List(x.size) { 1.0/x.size }
        return y.map { it/y.sum() }
    }

    private fun norm(x: List<Double>): List<Double> {
        val l = sqrt(x.asSequence().map { it*it }.sum())
        if (l == 0.0) return x
        return x.map { it / l }
    }

    override fun clone() = CNetwork().also { nw -> nw.layers.addAll(layers.map { it.clone() }) }

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