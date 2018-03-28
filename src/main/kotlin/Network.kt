interface Network {
    fun activate(x: Image): List<Double>
    fun clone(): Network
    val layers: MutableList<Layer>
}

class CNetwork(vararg layerSize: Int): Network {
    override val layers = MutableList(layerSize.size, { i -> Layer(layerSize[i]) })

    override fun activate(x: Image): List<Double> {
        var y = layers[0].cnn(x.colorsMatrix, 5, 1)
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

    override fun clone() = CNetwork().also { it.layers.addAll(layers.map { it.clone() }) }
}

class FNetwork(vararg layerSize: Int): Network {
    override val layers = MutableList(layerSize.size, { i -> Layer(layerSize[i]) })

    override fun activate(x: Image): List<Double> {
        var y = x.netOutputs.value
        layers.forEach {
            y = it.activate(y)
        }
        return Layer.softmax(y)
    }

    override fun clone() =  MNetwork().also {
        it.layers.addAll(layers.map { it.clone() })
    }
}

class MNetwork(vararg layerSize: Int): Network {
    override val layers = MutableList(layerSize.size, { i -> Layer(layerSize[i]) })

    override fun activate(x: Image): List<Double> {
        var y = x.netOutputs.value
        layers.forEach {
            y = it.activate(y)
        }
        return Layer.softmax(y)
    }

    override fun clone() =  MNetwork().also {
        it.layers.addAll(layers.map { it.clone() })
    }
}