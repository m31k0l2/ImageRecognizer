class Network(vararg layerSize: Int) {
    val layers = MutableList(layerSize.size, { i -> Layer(layerSize[i]) })

    fun activate(x: List<List<Double>>): List<Double> {
        var y = layers[0].cnn(x, 5, 1)
        y = y.map { Layer.relu(it) }
        y = y.map { Layer.pool(it) }
        y = layers[1].cnn(y, 5, 1)
        y = y.map { Layer.relu(it) }
        y = y.map { Layer.pool(it) }
        var o = y.flatMap { it }
        layers.subList(2, layers.size).forEach {
            o = it.activate(o)
        }
        return Layer.softmax(o)
    }

    fun clone() =  Network().also { it.layers.addAll(layers.map { it.clone() }) }
}