interface Network {
    fun activate(x: Image): List<Double>
    fun clone(): Network
    val layers: MutableList<Layer>
}

class CNetwork(vararg layerSize: Int): Network {
    override val layers = MutableList(layerSize.size, { i -> Layer(layerSize[i]) })

    companion object {
        val dividers = listOf(
                MatrixDivider(28,5,1),
                MatrixDivider(24,2,2),
                MatrixDivider(12,5,1),
                MatrixDivider(8,2,2),
                MatrixDivider(4,2,1),
                MatrixDivider(2,2,1)
        )
    }

    override fun activate(x: Image): List<Double> {
        var y = layers[0].cnn(x.colorsMatrix, dividers[0])
        y = y.map { Layer.relu(it) }
        y = y.map { Layer.pool(it, dividers[1]) }
        y = layers[1].cnn(y, dividers[2])
        y = y.map { Layer.relu(it) }
        y = y.map { Layer.pool(it, dividers[3]) }
        y = layers[2].cnn(y, dividers[4])
        y = y.map { Layer.relu(it) }
        y = layers[3].cnn(y, dividers[5])
        var o = y.flatMap { it }
        layers.subList(4, layers.size).forEach {
            o = it.activate(o)
        }
        return Layer.softmax(o)
    }

    override fun clone() = CNetwork().also { it.layers.addAll(layers.map { it.clone() }) }
}