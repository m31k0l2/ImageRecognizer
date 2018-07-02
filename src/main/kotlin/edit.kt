import kotlin.math.round

fun main(args: Array<String>) {
    var nw0 = CNetwork().load("nets/nw.net")!!
    val batch = MNIST.buildBatch(500)
    val r0 = nw0.rate(6, batch)
    nw0 = nw0.rebuild(6, MNIST.buildBatch(500))
    val r1 = nw0.rate(6, batch)
    nw0.save("nets/null.net")
    println("$r0 -> $r1")
    println(getStructure("nets/null.net")?.toList())
}

fun Network.rebuild(rateNumber: Int, batch: Set<Image>): Network {
    val r0 = rate(rateNumber, batch)
    if (r0 < 0.2) {
        log.info("$r0")
        return this
    }
    val nw = clearCNNLayers(rateNumber, batch).clearFNNLayers(4, rateNumber, batch)
    val r1 = nw.rate(rateNumber, batch)
    log.info("rebuild: $r0 -> $r1")
    log.info("new structure: ${nw.layers.map { it.neurons.size }}")
    return nw
}

fun Network.clearFNNLayers(layerPosition: Int, rateNumber: Int, batch: Set<Image>): Network {
    var nw0 = this
    var r0 = nw0.rate(rateNumber, batch)
    for (neuronPosition in layers[layerPosition].neurons.size-1 downTo 0) {
        val nw = nw0.clone()
        val clearLayer = nw.layers[layerPosition]
        val nextLayer = nw.layers[layerPosition+1]
        clearLayer.neurons.removeAt(neuronPosition)
        nextLayer.neurons.forEach { it.weights.removeAt(neuronPosition + 1) }
        val rate = nw.rate(rateNumber, batch)
        if (rate >= r0) {
            nw0 = nw
            r0 = rate
        }
    }
    return nw0
}

fun Network.clearCNNLayers(rateNumber: Int, batch: Set<Image>): Network {
    var nw0 = this
    val r0 = nw0.rate(rateNumber, batch)
    if (r0 < 0.1) return this
    for (l in 0..3) {
        for (i in nw0.layers[l].neurons.size - 1 downTo 0) {
            val nw = nw0.clone()
            nw.removeNeuronFromLayer(l, i)
            val rate = nw.rate(rateNumber, batch)
            if (round(rate * 100) >= round(r0 * 100)) {
                nw0 = nw
            }
        }
    }
    return nw0
}

fun Network.removeNeuronFromLayer(layerPosition: Int, neuronPosition: Int) {
    val count = layers[layerPosition].neurons.size
    val size = if (layerPosition < 3) layers.subList(layerPosition+1, 4).map { it.neurons.size }.reduce { acc, i -> acc*i } else count
    layers[4].neurons.forEach { neuron ->
        val weights = neuron.weights.subList(1, neuron.weights.size).chunked(size).toMutableList()
        for (i in weights.size-1 downTo 0 ) {
            if (i % count == neuronPosition) weights.removeAt(i)
        }
        neuron.weights = listOf(neuron.weights.first(), *weights.flatten().toTypedArray()).toMutableList()
    }
    layers[layerPosition].neurons.removeAt(neuronPosition)
}