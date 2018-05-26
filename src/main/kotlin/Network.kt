import java.util.*

interface Network {
    fun activate(x: Image): List<Double>
    fun clone(): Network
    val layers: MutableList<Layer>
    fun dropout(layersNumbers: List<Int>, dropoutRate: Double, isRandom: Boolean=false): Network
}

class CNetwork(vararg layerSize: Int): Network {
    override val layers = MutableList(layerSize.size, { i -> Layer(layerSize[i]) })
    private val calcImages = mutableMapOf<Image, List<Double>>()

    companion object {
        val dividers = listOf(
                MatrixDivider(28,5,1),
                MatrixDivider(24,2,2),
                MatrixDivider(12,5,1),
                MatrixDivider(8,2,2),
                MatrixDivider(4,2,1),
                MatrixDivider(2,2,1)
        )
        var teachFromLayer = 0
    }

    private fun activateLayer0(x: List<List<Double>>): List<List<Double>> {
        var y = layers[0].cnn(x, dividers[0])
        y = y.map { Layer.relu(it) }
        return y.map { Layer.pool(it, dividers[1]) }
    }

    private fun activateLayer1(x: List<List<Double>>): List<List<Double>> {
        var y = layers[1].cnn(x, dividers[2])
        y = y.map { Layer.relu(it) }
        return y.map { Layer.pool(it, dividers[3]) }
    }

    private fun activateLayer2(x: List<List<Double>>): List<List<Double>> {
        val y = layers[2].cnn(x, dividers[4])
        return y.map { Layer.relu(it) }
    }

    private fun activateLayer3(x: List<List<Double>>): List<List<Double>> {
        return layers[3].cnn(x, dividers[5])
    }

    private fun activate2layers(x: Image): List<Double> {
        if (calcImages.containsKey(x)) return calcImages[x]!!
        val y = activateLayer0(x.colorsMatrix)
        val o = layers[1].activate(y.flatten())
        val result = Layer.softmax(o)
        calcImages[x] = result
        return result
    }

    private fun activate3layers(x: Image): List<Double> {
        if (calcImages.containsKey(x)) return calcImages[x]!!
        var y: List<List<Double>>
        if (x.y == null) {
            y = activateLayer0(x.colorsMatrix)
            if (teachFromLayer == 1) x.y = y
            y = activateLayer1(y)
        } else {
            y = activateLayer1(x.y!!)
        }
        val o = layers[2].activate(y.flatten())
        val result = Layer.softmax(o)
        calcImages[x] = result
        return result
    }

    private fun activate4layers(x: Image): List<Double> {
        if (calcImages.containsKey(x)) return calcImages[x]!!
        var o: List<Double>
        if (x.o == null) {
            var y: List<List<Double>>
            if (x.y == null) {
                y = activateLayer0(x.colorsMatrix)
                if (teachFromLayer == 1) x.y = y
                y = activateLayer1(y)
                if (teachFromLayer == 2) x.y = y
                y = activateLayer2(y)
            } else {
                when (teachFromLayer) {
                    1 -> {
                        y = activateLayer1(x.y!!)
                        y = activateLayer2(y)
                    }
                    else -> y = activateLayer2(x.y!!)
                }
            }
            o = y.flatMap { it }
            if (teachFromLayer == 2) x.o = o
        } else {
            o = x.o!!
        }
        o = layers[3].activate(o)
        val result = Layer.softmax(o)
        calcImages[x] = result
        return result
    }

    private fun activate6layers(x: Image): List<Double> {
        if (calcImages.containsKey(x)) return calcImages[x]!!
        var o: List<Double>
        if (x.o == null) {
            var y: List<List<Double>>
            if (x.y == null) {
                y = activateLayer0(x.colorsMatrix)
                if (teachFromLayer == 1) x.y = y
                y = activateLayer1(y)
                if (teachFromLayer == 2) x.y = y
                y = activateLayer2(y)
                if (teachFromLayer == 3) x.y = y
                y = activateLayer3(y)
            } else {
                when (teachFromLayer) {
                    1 -> {
                        y = activateLayer1(x.y!!)
                        y = activateLayer2(y)
                        y = activateLayer3(y)
                    }
                    2 -> {
                        y = activateLayer2(x.y!!)
                        y = activateLayer3(y)
                    }
                    else -> y = activateLayer3(x.y!!)
                }
            }
            o = y.flatMap { it }
            o = Layer.relu(o)
            o = Layer.norm(o)
            if (teachFromLayer == 4) x.o = o
            o = layers[4].activate(o)
            if (teachFromLayer == 5) x.o = o
        } else {
            o = x.o!!
            if (teachFromLayer == 4) o = layers[4].activateSigma(o)
        }
        o = layers[5].activateSigma(o)
        val result = Layer.softmax(o)
        calcImages[x] = result
        return result
    }

    private fun activate5layers(x: Image): List<Double> {
        if (calcImages.containsKey(x)) return calcImages[x]!!
        var o: List<Double>
        if (x.o == null) {
            var y: List<List<Double>>
            if (x.y == null) {
                y = activateLayer0(x.colorsMatrix)
                if (teachFromLayer == 1) x.y = y
                y = activateLayer1(y)
                if (teachFromLayer == 2) x.y = y
                y = activateLayer2(y)
                if (teachFromLayer == 3) x.y = y
                y = activateLayer3(y)
            } else {
                when (teachFromLayer) {
                    1 -> {
                        y = activateLayer1(x.y!!)
                        y = activateLayer2(y)
                        y = activateLayer3(y)
                    }
                    2 -> {
                        y = activateLayer2(x.y!!)
                        y = activateLayer3(y)
                    }
                    else -> y = activateLayer3(x.y!!)
                }
            }
            o = y.flatMap { it }
            if (teachFromLayer == 4) x.o = o
        } else {
            o = x.o!!
        }
        o = layers[4].activate(o)
        val result = Layer.softmax(o)
        calcImages[x] = result
        return result
    }

    private fun activateCNNLayers(x: Image): List<Double> {
        if (calcImages.containsKey(x)) return calcImages[x]!!
        var y: List<List<Double>>
        if (x.y == null) {
            y = activateLayer0(x.colorsMatrix)
            if (teachFromLayer == 1) x.y = y
            y = activateLayer1(y)
            if (teachFromLayer == 2) x.y = y
            y = activateLayer2(y)
            if (teachFromLayer == 3) x.y = y
            y = activateLayer3(y)
        } else {
            when (teachFromLayer) {
                1 -> {
                    y = activateLayer1(x.y!!)
                    y = activateLayer2(y)
                    y = activateLayer3(y)
                }
                2 -> {
                    y = activateLayer2(x.y!!)
                    y = activateLayer3(y)
                }
                else -> y = activateLayer3(x.y!!)
            }
        }
        val result = Layer.softmax(y.flatten())
        calcImages[x] = result
        return result
    }

    override fun activate(x: Image): List<Double> {
        layers.find { it.neurons.first().weights.size == 0 }?.let {
            x.o = null
            x.y = null
        }
        return activate6layers(x)
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