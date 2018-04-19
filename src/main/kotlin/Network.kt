interface Network {
    fun activate(x: Image): List<Double>
    fun clone(): Network
    val layers: MutableList<Layer>
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

    override fun activate(x: Image): List<Double> {
        if (calcImages.containsKey(x)) return calcImages[x]!!
        var o: List<Double>
        if (x.o == null) {
            var y = if (teachFromLayer > 0 && x.y != null) x.y!! else activateLayer0(x.colorsMatrix)
            if (x.y == null && teachFromLayer == 1) x.y = y
            if (x.y == null || teachFromLayer <= 1) y = activateLayer1(y)
            if (x.y == null && teachFromLayer == 2) x.y = y
            if (x.y == null || teachFromLayer <= 2) y = activateLayer2(y)
            if (x.y == null && teachFromLayer == 3) x.y = y
            if (x.y == null || teachFromLayer <= 3) y = activateLayer3(y)
            if (x.y == null && teachFromLayer == 4) x.y = y
            o = y.flatMap { it }
        } else {
            o = x.o!!
        }
        if (x.o == null && teachFromLayer == 4) x.o = o
        if (x.o == null || teachFromLayer <= 4) o = layers[4].activate(o)
        if (x.o == null && teachFromLayer == 5) x.o = o
        if (x.o == null || teachFromLayer <= 5) o = layers[5].activate(o)
//        if (x.o == null && teachFromLayer == 6) x.o = o
//        if (x.o == null || teachFromLayer <= 6) o = layers[6].activate(o)
        val result = Layer.softmax(o)
        calcImages[x] = result
        return result
    }

    override fun clone() = CNetwork().also { it.layers.addAll(layers.map { it.clone() }) }
}